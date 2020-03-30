
from .run_squad import process_inputs, process_result, process_output
import grpc  
import json
import os
import requests
from . import tokenization
import grpc 
import tensorflow as tf
from tensorflow_serving.apis import prediction_service_pb2_grpc, predict_pb2

tf1 = tf.compat.v1

class Client(object):  

    def __init__(self, hostport): 
        self.hostport = hostport
        self.headers = { "Content-type": "application/json" }
        self.channel = grpc.insecure_channel(self.hostport)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(self.channel)
        self.model_request = predict_pb2.PredictRequest() 

    def predict(self, input_data, _id):
        
        self.generate_test_file(input_data, _id) 
        self.process_inputs(_id) 

        record_iterator = tf1.python_io.tf_record_iterator(path='./app/inputs/{}/eval.tf_record'.format(_id))

        self.model_request.model_spec.name = 'bert-qa'

        all_results = []

        for string_record in record_iterator:
            
            self.model_request.inputs['examples'].CopyFrom(
                    tf.make_tensor_proto(
                        string_record,
                        dtype=tf.string,
                        shape=[1]
                    )
            )
            
            result_future = self.stub.Predict.future(self.model_request, 30.0)  
            raw_result = result_future.result().outputs
            all_results.append(process_result(raw_result))

        result = process_output(all_results, self.examples, self.features, input_data, _id)
        return json.dumps(result)

    def process_inputs(self, _id):
        self.examples, self.features = process_inputs(_id)


    def generate_test_file(self, input_data, _id): 
        with open('./app/inputs/{}/input.json'.format(_id), 'w') as outfile:
            json.dump(input_data, outfile) 