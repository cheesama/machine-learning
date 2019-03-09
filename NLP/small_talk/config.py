#-*- coding: utf-8 -*-
import tensorflow as tf

tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch_size')
tf.app.flags.DEFINE_integer('train_steps', 10000, 'train steps') #train epoch
tf.app.flags.DEFINE_float('dropout_width', 0.5, 'dropout width') #dropout ratio
tf.app.flags.DEFINE_integer('layer_size', 3, 'layer size')
tf.app.flags.DEFINE_integer('hidden_size', 128, 'hidden size')
tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
tf.app.flags.DEFINE_string('data_path', 'data_in/ChatBotData.csv', 'data path')
tf.app.flags.DEFINE_string('vocabulary_path', 'data_out/vocabularyData.voc', 'vocabulary path')
tf.app.flags.DEFINE_string('check_point_path', 'data_out/check_point', 'checkpoint path')
tf.app.flags.DEFINE_integer('shuffle_seek', 1000, 'shuffle random seed')
tf.app.flags.DEFINE_integer('max_sequence_length', 25, 'max sequence length')
tf.app.flags.DEFINE_integer('embedding_size', 128, 'embedding size')
tf.app.flags.DEFINE_boolean('tokenize_as_morph', True, 'set morph tokenize')
tf.app.flags.DEFINE_boolean('embedding', True, 'Use Embedding flag')
tf.app.flags.DEFINE_boolean('multilayer', True, 'Use Multi RNN Cell')

#Define FLAGS
DEFINES = tf.app.flags.FLAGS

