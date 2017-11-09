#!/usr/bin/env python

import os
import tensorflow as tf
from tensorflow.contrib import lookup
import ner_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_path', '/home/fanyuguang/Project/tensorflow/tensorflow_serving/example/ner-checkpoint/', 'checkpoint directory')
tf.app.flags.DEFINE_string('export_path', '/home/fanyuguang/Project/tensorflow/tensorflow_serving/example/ner-export/', 'Directory where to export inference model')
tf.app.flags.DEFINE_string('vocab_path', '/home/fanyuguang/Project/tensorflow/tensorflow_serving/example/ner-vocab/', 'vocab directory')

tf.app.flags.DEFINE_integer('num_steps', 50, 'num steps, equals the length of words')
tf.app.flags.DEFINE_integer('vocab_size', 200000, 'vocab size')
tf.app.flags.DEFINE_integer('embedding_size', 100, 'word embedding size')
tf.app.flags.DEFINE_integer('hidden_size', 100, 'lstm hidden size')
tf.app.flags.DEFINE_float('keep_prob', 0.5, 'keep prob')
tf.app.flags.DEFINE_integer('num_layers', 2, 'lstm layers')
tf.app.flags.DEFINE_integer('num_classes', 9, 'named entity classes')
tf.app.flags.DEFINE_float('prop_limit', 0.99, 'limit predict prop')


def export():
  checkpoint_path = FLAGS.checkpoint_path
  export_path = FLAGS.export_path
  vocab_path = FLAGS.vocab_path

  num_steps = FLAGS.num_steps
  vocab_size = FLAGS.vocab_size
  embedding_size = FLAGS.embedding_size
  hidden_size = FLAGS.hidden_size
  keep_prob = FLAGS.keep_prob
  num_layers = FLAGS.num_layers
  num_classes = FLAGS.num_classes
  prop_limit = FLAGS.prop_limit

  # split 1-D String dense Tensor to words SparseTensor
  sentences = tf.placeholder(dtype=tf.string, shape=[None], name='input_sentences')
  sparse_words = tf.string_split(sentences, delimiter=' ')

  # slice SparseTensor
  valid_indices = tf.less(sparse_words.indices, tf.constant([num_steps], dtype=tf.int64))
  valid_indices = tf.reshape(tf.split(valid_indices, [1, 1], axis=1)[1], [-1])
  valid_sparse_words = tf.sparse_retain(sparse_words, valid_indices)

  excess_indices = tf.greater_equal(sparse_words.indices, tf.constant([num_steps], dtype=tf.int64))
  excess_indices = tf.reshape(tf.split(excess_indices, [1, 1], axis=1)[1], [-1])
  excess_sparse_words = tf.sparse_retain(sparse_words, excess_indices)

  # sparse to dense
  words = tf.sparse_to_dense(sparse_indices=valid_sparse_words.indices,
                             output_shape=[valid_sparse_words.dense_shape[0], num_steps],
                             sparse_values=valid_sparse_words.values,
                             default_value='_PAD')

  # dict words to token ids
  words_table = lookup.index_table_from_file(os.path.join(vocab_path, 'words_vocab.txt'), default_value=3)
  words_ids = words_table.lookup(words)

  # blstm model predict
  with tf.variable_scope('model', reuse=None):
    logits, _ = ner_model.inference(words_ids, valid_sparse_words.dense_shape[0], num_steps, vocab_size, embedding_size, hidden_size,
                                 keep_prob, num_layers, num_classes, is_training=False)
  props = tf.nn.softmax(logits)
  max_prop_values, max_prop_indices = tf.nn.top_k(props, k=1)

  predict_scores = tf.reshape(max_prop_values, shape=[-1, num_steps])
  predict_labels_ids = tf.reshape(max_prop_indices, shape=[-1, num_steps])
  predict_labels_ids = tf.to_int64(predict_labels_ids)

  # replace untrusted prop that less than prop_limit
  trusted_prop_flag = tf.greater_equal(predict_scores, tf.constant(prop_limit, dtype=tf.float32))
  replace_prop_labels_ids = tf.to_int64(tf.fill(tf.shape(predict_labels_ids), 4))
  predict_labels_ids = tf.where(trusted_prop_flag, predict_labels_ids, replace_prop_labels_ids)

  # dict token ids to labels
  labels_table = lookup.index_to_string_table_from_file(os.path.join(vocab_path, 'labels_vocab.txt'), default_value='o')
  predict_labels = labels_table.lookup(predict_labels_ids)

  # extract real blstm predict label in dense and save to sparse
  valid_sparse_predict_labels = tf.SparseTensor(indices=valid_sparse_words.indices,
                                                values=tf.gather_nd(predict_labels, valid_sparse_words.indices),
                                                dense_shape=valid_sparse_words.dense_shape)

  # create excess label SparseTensor with 'O'
  excess_sparse_predict_labels = tf.SparseTensor(indices=excess_sparse_words.indices,
                                                 values=tf.fill(tf.shape(excess_sparse_words.values), 'O'),
                                                 dense_shape=excess_sparse_words.dense_shape)

  # concat SparseTensor
  sparse_predict_labels = tf.SparseTensor(indices=tf.concat(axis=0, values=[valid_sparse_predict_labels.indices, excess_sparse_predict_labels.indices]),
                                          values=tf.concat(axis=0, values=[valid_sparse_predict_labels.values, excess_sparse_predict_labels.values]),
                                          dense_shape=excess_sparse_predict_labels.dense_shape)
  sparse_predict_labels = tf.sparse_reorder(sparse_predict_labels)

  # join SparseTensor to 1-D String dense Tensor
  # remain issue, num_split should equal the real size, but here limit to 1
  join_labels_list = []
  slice_labels_list = tf.sparse_split(sp_input=sparse_predict_labels, num_split=1, axis=0)
  for slice_labels in slice_labels_list:
    slice_labels = slice_labels.values
    join_labels = tf.reduce_join(slice_labels, reduction_indices=0, separator=' ')
    join_labels_list.append(join_labels)
  format_predict_labels = tf.stack(join_labels_list)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      print('read model from {}'.format(ckpt.model_checkpoint_path))
      saver.restore(sess, ckpt.model_checkpoint_path)
      global_step = int(ckpt.model_checkpoint_path.split('-')[-1])
    else:
      print('No checkpoint file found at %s' % FLAGS.checkpoint_path)
      return

    # Export inference model.
    output_path = os.path.join(export_path, str(global_step))
    print 'Exporting trained model to', output_path
    builder = tf.saved_model.builder.SavedModelBuilder(output_path)

    # Build the signature_def_map.
    predict_inputs_tensor_info = tf.saved_model.utils.build_tensor_info(sentences)
    predict_output_tensor_info = tf.saved_model.utils.build_tensor_info(format_predict_labels)
    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
      inputs={
        'input_sentences': predict_inputs_tensor_info,
      },
      outputs={'classes': predict_output_tensor_info},
      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={'predict_ner': prediction_signature},
      legacy_init_op=legacy_init_op)

    builder.save()
    print 'Successfully exported model to %s' % export_path


def main(_):
  export()


if __name__ == '__main__':
  tf.app.run()
