from bert_serving.client import BertClient
import numpy as np
import json
import time
from sentence_transformers import models, SentenceTransformer

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

word_embedding_model = models.BERT('./app/bert_model')

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

bert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

def get_categories_questions():
    '''
        Retorna uma lista de perguntas 
    '''

    with open("./app/covid-final-train.json") as file:
        dataset = json.load(file)


    results = {}

    for index, item in enumerate(dataset["data"]): 
        data = dataset["data"][index]['paragraphs']
        
        pre_list = []
        for item in data:
            pre_list.append(item['qas'])

        questions = []

        for i in range(len(pre_list)):
            for j in range(len(pre_list[i])):
                questions.append(pre_list[i][j]['question'])

        
        results[number_to_category[index]] = questions

    return results


def get_context_dict(category): 
    ''' 
        Retorna um dicionário mapeando contextos à uma lista perguntas
    '''

    with open("./app/covid-final-train.json") as train_file:
        dataset = json.load(train_file)
        
    for index, item in enumerate(dataset["data"]): 
        data = dataset["data"][category_to_number[category]]['paragraphs']
        

    context_to_questions = {}
    
    for i in range(len(data)):
        context_to_questions[data[i]['context']] = []
        for j in range(len(data[i]['qas'])):
            context_to_questions[data[i]['context']].append(data[i]['qas'][j]['question'])
            
    return context_to_questions


def get_similarity(category, question):
    # Ambos virão da request
    # Inicialização de variáveis
    #json.dumps(bc.server_status, ensure_ascii=Fals   
    topk = 3
    
    query_vec = bert_model.encode([question])[0]

    # "tópico" e "query" vão chegar pela request

    category_to_questions = get_categories_questions()

    questions = category_to_questions[category]


    context_to_questions = get_context_dict(category)


    doc_vecs = bert_model.encode(questions)


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
        return context
                
        # print("Número de contextos únicos encontrados: {}".format(contexts))
        # print("Número de contextos: {}".format(len(contexts)))

