import tensorflow as tf
#filter dan weights sama saja. jika di fully connected network disebut weight, kalau di CNN
#disebut filter
def new_weights(shape,name):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05, dtype=tf.float32),dtype=tf.float32,name='weight_' + name)

def new_biases(length,name):
    return tf.Variable(tf.constant(0.05, shape=[length], dtype=tf.float32), dtype=tf.float32, name='bias_' + name)

def new_fc_layer(input, num_inputs, num_outputs, name, activation="RELU"):
    weights = new_weights(shape=[num_inputs,num_outputs],name=name)
    biases = new_biases(length=num_outputs,name=name)
    layer = tf.matmul(input,weights) + biases

    if activation == "RELU":
        layer = tf.nn.relu(layer)
    elif activation == "LRELU":
        layer = tf.nn.leaky_relu(layer)
    elif activation == "SELU":
        layer = tf.nn.selu(layer)
    elif activation == "ELU":
        layer = tf.nn.elu(layer)
    elif activation == "SIGMOID":
        layer = tf.nn.sigmoid(layer)
    elif activation == "SOFTMAX":
        layer == tf.nn.softmax(layer)
    return layer, weights, biases

def new_conv_layer(input, filter_shape, name, activation == "RELU", padding = "SAME", strides=[1,1,1,1]):
    shape = filter_shape
    weights = new_weights(shape=shape,name=name)
    biases = new_biases(length=filter_shape[3],name=name)

    layer = tf.nn.conv2d(input=input,filter=weights,strides=strides,padding=padding,name='convolution_' + name)

    layer += biases
    if activation == "RELU":
        layer = tf.nn.relu(layer)
    elif activation == "LRELU":
        layer = tf.nn.leaky_relu(layer)
    elif activation == "SELU":
        layer = tf.nn.selu(layer)
    elif activation == "ELU":
        layer = tf.nn.elu(layer)
    elif activation == "SIGMOID":
        layer = tf.nn.sigmoid(layer)
    elif activation == "SOFTMAX":
        layer == tf.nn.softmax(layer)
    return layer, weights, biases

def new_deconv_layer(input, filter_shape, output_shape, name, activation = 'RELU', strides = [1,1,1,1], padding = 'SAME'):
    weights = tf.Variable(tf.truncated_normal(filter_shape[0],filter_shape[1], filter_shape[3], filter_shape[2]))
    biases = new biases(length=filter_shape[3],name=name)
    deconv_shape = tf.stack(output_shape)
    deconv = tf.nn.conv2d_transpose(input=input)