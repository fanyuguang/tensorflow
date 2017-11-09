#!/bin/sh

# ../bazel-bin/tensorflow_serving/example/ner_saved_model
CUDA_VISIBLE_DEVICES='' ../bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=8500 --model_name=ner --model_base_path=../tensorflow_serving/example/ner-export/
