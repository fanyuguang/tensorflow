#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('server', 'localhost:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('sentence', '范冰冰 在 娱乐圈 拥有 很多 粉丝', 'Predict sentence')

def main(_):
  host, port = FLAGS.server.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'ner'
  request.model_spec.signature_name = 'predict_ner'
  sentence = FLAGS.sentence
  request.inputs['input_sentences'].CopyFrom(tf.contrib.util.make_tensor_proto([sentence], shape=[1], dtype=tf.string))
  result = stub.Predict(request, 10.0)  # 10 secs timeout
  # print result
  print result.outputs['classes'].string_val


if __name__ == '__main__':
  tf.app.run()
