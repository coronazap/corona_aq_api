#!/usr/bin/python
# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify
import os 
import json
from multiprocessing import Process, Queue
from bert-client.bert_client import Client
import uuid
from bert-client.run_squad import process_inputs

app = Flask(__name__) 

hostport =  os.getenv('BERT_HOSTPORT')

client = Client(hostport)

@app.route('/api/predict', methods=['POST'])
def predict():

    data = request.get_json()
    
    input_data = {
        "data": [
            {
                "title": data['category'],
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "question": data['question'],
                                "id": str(uuid.uuid1()),
                                "is_impossible": ""
                            },

                        ],
                        "context": data['context']}
                ]
            }
        ]
    }

    results = client.predict(input_data)

    return jsonify(results)


@app.route('/api/similarity', methods=['POST']) 
def get_context(): 

    pass

if __name__ == '__main__':
    app.run(debug=True)

