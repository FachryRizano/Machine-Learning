import tensorflow as tf
#filter dan weights sama saja. jika di fully connected network disebut weight, kalau di CNN
#disebut filter
def new_weights(shape,name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05, dtype=tf.float32),dtype=tf.float32,name='weight_' + name)

def new_biases(length,name):
    return tf.Variable(tf.constant(0.05, shape=[length], dtype=tf.float32), dtype=tf.float32, name='bias_' + name)

def new_fc_layer(input, num_inputs, num_outputs, name, activation="RELU"):
    weights = new_weights(shape=[num_inputs,num_outputs],name=name)
    biases = new_biases(shape)