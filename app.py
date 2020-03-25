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

client = Client(hostport)

@app.route('/api/predict', methods=['POST'])
def predict():

    
    data = request.get_json()

    get_similarity(data['question'], data['category'])
   
    input_data = {
        "version": "1,1",
        "data": [
            {
                "title": data['category'],
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "question": "Como se da a transmissao do COVID0-19?",
                                "id": str(uuid.uuid1()),
                                "is_impossible": ""
                            },
                        ],
                        "context": "As investigações sobre as formas de transmissão do coronavírus ainda estão em andamento, mas a disseminação de pessoa para pessoa, ou seja, a contaminação por gotículas respiratórias ou contato, está ocorrendo. Qualquer pessoa que tenha contato próximo (cerca de 1m) com alguém com sintomas respiratórios está em risco de ser exposta à infecção. A transmissão dos coronavírus costuma ocorrer pelo ar ou por contato pessoal com secreções contaminadas, como: gotículas de saliva; espirro; tosse; catarro; contato pessoal próximo, como toque ou aperto de mão; contato com objetos ou superfícies contaminadas, seguido de contato com a boca, nariz ou olhos."
                    }
                ]
            }
        ]
    }

    results = client.predict(input_data)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)

