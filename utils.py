#from bert_serving.client import BertClient
#import numpy as np
import json
import time

category_to_number = {
    'prevencao': 0,
    'sintomas': 1,
    'transmissao': 2,
    'tratamento': 3
}

number_to_category = {
    0: 'prevencao',
    1: 'sintomas',
    2: 'transmissao', 
    3: 'tratamento'
}


def get_categories_questions():
    '''
        Retorna uma lista de perguntas 
    '''

    with open("./covid-final-train.json") as file:
        dataset = json.load(file)


    results = {}

    for index, item in enumerate(dataset["data"]): 
        data = dataset["data"][index]['paragraphs']
        
        pre_list = []
        for item in data:
            print(item)
            pre_list.append(item['qas'])

        questions = []

        for i in range(len(pre_list)):
            for j in range(len(pre_list[i])):
                questions.append(pre_list[i][j]['question'])

        
        results[number_to_category[index]] = questions

    print(results)

    return results


def get_context_dict(): 
    ''' 
        Retorna um dicionário mapeando contextos à uma lista perguntas
    '''

    with open("./covid-final-train.json") as train_file:
        dataset = json.load(train_file)
        
    for index, item in enumerate(dataset["data"]): 
        data = dataset["data"][index]['paragraphs']
        

    context_to_questions = {}
    
    for i in range(len(data)):
        context_to_questions[data[i]['context']] = []
        for j in range(len(data[i]['qas'])):
            context_to_questions[data[i]['context']].append(data[i]['qas'][j]['question'])
            
    return context_to_questions

context_to_questions = get_context_dict()
category_to_questions = get_categories_questions()

def get_similarity(category, question):

    # Ambos virão da request
    # Inicialização de variáveis
    bc = BertClient(port=5555, port_out=5556)
    json.dumps(bc.server_status, ensure_ascii=False)
    topk = 3
    query_vec = bc.encode([question])[0]

    # "tópico" e "query" vão chegar pela request

    questions = category_to_questions[category]

    doc_vecs = bc.encode(questions)


    # Caso seja apenas a pergunta #1 (top pergunta)
    #---------------------------------------------------
    if topk == 1:
        score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        topQuestion = questions[max(topk_idx)]

        for key in context_to_questions.keys():
            if topQuestion in context_to_questions[key]:
                context = key
                return format(context)
    #------------------------------------------------------


    # Caso sejam N top perguntas
    #-------------------------------------------------------
    if topk > 1:
        score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        topQuestions = []
        # contexts = set()
        contexts = []

        for idx in topk_idx:
            topQuestions.append(questions[idx])

        for key in context_to_questions.keys():
            
            for topQuestion in topQuestions:
                
                if topQuestion in context_to_questions[key]:
                    contexts.append(key)
                                        
        context = max(set(contexts), key=contexts.count)
        return format(context)
                
        # print("Número de contextos únicos encontrados: {}".format(contexts))
        # print("Número de contextos: {}".format(len(contexts)))

