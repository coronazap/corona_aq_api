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
# from utils import get_similarity

app = Flask(__name__) 

hostport =  os.getenv('BERT_HOSTPORT')

global client 
client = Client(hostport)

global contexts 
contexts = {}

with open('./app/covid-final-train.json') as contexts_file: 
    contexts_data = json.load(contexts_file)
    contexts['prevencao'] = contexts_data['data'][0]['paragraphs'][0]['context']
    contexts['sintomas'] = contexts_data['data'][1]['paragraphs'][0]['context']
    contexts['transmissao'] = contexts_data['data'][2]['paragraphs'][0]['context']
    contexts['tratamento'] = contexts_data['data'][3]['paragraphs'][0]['context']


@app.route('/api/predict', methods=['POST'])
def predict():


    data = request.get_json()

    context = contexts[data['category']]
    _id = str(uuid.uuid1())

    # context = get_similarity(data["category"], data["question"])

    os.mkdir('./app/results/{}'.format(_id))
    os.mkdir('./app/inputs/{}'.format(_id))
        
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
                                "id": _id,
                                "is_impossible": ""
                            },
                        ],
                        "context": context
                    }
                ]
            }
        ]
    }

    results = client.predict(input_data, _id)

    n_best_predictions = json.loads(results)[0]['n_best_predictions']

    answers = []

    for answer in n_best_predictions:
            answers.append(answer['text']) 

    return {
        'answer': str(answers[0]).capitalize()
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

