#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os 
import json
from multiprocessing import Process, Queue
from bert_client.main import Client
import uuid
from bert_client.run_squad import process_inputs
import time 
from utils import get_similarity

app = Flask(__name__) 

hostport =  os.getenv('BERT_HOSTPORT')

global client 
global contexts 

contexts = {}

with open('./app/covid-final-train.json') as contexts_file: 
    contexts['prevencao'] = contexts_file['data'][0]['paragraphs'][0]['context']
    contexts['sintomas'] = contexts_file['data'][1]['paragraphs'][0]['context']
    contexts['transmissao'] = contexts_file['data'][2]['paragraphs'][0]['context']
    contexts['tratamento'] = contexts_file['data'][3]['paragraphs'][0]['context']

client = Client(hostport)

@app.route('/api/predict', methods=['POST'])
def predict():

    data = request.get_json()

    context = contexts[question['category']]

    # context = get_similarity(data["category"], data["question"])
    
    print(' ')
    print('Question: ' + data['question'])
    print(' ')
    print('Chosen context: ' + context])
    print(' ')
    
    input_data = {
        "version": "1,1",
        "data": [
            {
                "title": data['category'],
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "question": data["question"],
                                "id": str(uuid.uuid1()),
                                "is_impossible": ""
                            },
                        ],
                        "context": context
                    }
                ]
            }
        ]
    }

    results = client.predict(input_data)

    n_best_predictions = json.loads(results)[0]['n_best_predictions']

    answers = []

    for answer in n_best_predictions:
            answers.append(answer['text']) 

    print(' ')
    print('Resposta: ' + str(answers))

    return str(answers)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

