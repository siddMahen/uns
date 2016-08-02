import tensorflow as tf
import numpy as np

def variable_summaries(var, name):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def weight_var(name, shape, wd=None):
    stddev = np.sqrt(2.0/np.prod(shape[:-1]))
    var = _variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))

    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var

def bias_var(name, shape):
    var = _variable_on_cpu(name, shape, tf.constant_initializer(0.01))
    return var

def conv2d(x, W, s):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

def max_pool(x, k, s):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
            padding='SAME')

def conv_layer(input_tensor, input_dim, output_dim, k, s, layer_name):
    with tf.variable_scope(layer_name):
        weights = weight_var("weights", [k, k, input_dim, output_dim])
        bias = bias_var("bias", [output_dim])

        variable_summaries(weights, layer_name + '/weights')
        variable_summaries(bias, layer_name + '/biases')

        activations = conv2d(input_tensor, weights, s) + bias
        relu = tf.nn.relu(activations, 'relu')

        tf.histogram_summary(layer_name + '/activations', activations)
        tf.histogram_summary(layer_name + '/activations_relu', relu)

    return relu

def fc_layer(input_tensor, input_dim, output_dim, keep_prob, layer_name, final=False):
    with tf.variable_scope(layer_name):
        weights = weight_var("weights", [input_dim, output_dim])
        bias = bias_var("bias", [output_dim])

        variable_summaries(weights, layer_name + '/weights')
        variable_summaries(bias, layer_name + '/biases')

        activations = tf.matmul(input_tensor, weights) + bias
        tf.histogram_summary(layer_name + '/activations', activations)

        if final == True:
            return activations
        else:
            relu = tf.nn.relu(activations, 'relu')
            tf.histogram_summary(layer_name + '/activations_relu', relu)
            return tf.nn.dropout(relu, keep_prob)

def inference(images, keep_prob):
    # new, inception based architecture
    with tf.variable_scope("conv1"):
        c1 = conv_layer(images, 1, 64, k=7, s=2, layer_name="conv1")
        mp1 = max_pool(c1, k=3, s=2)
        lrn1 = tf.nn.lrn(mp1, depth_radius=2, bias=1.0, alpha=1e-3, beta=0.75)

    with tf.variable_scope("conv2"):
        c2_1x1 = conv_layer(lrn1, 64, 64, k=1, s=1, layer_name="conv2_1x1_pre")
        c2 = conv_layer(c2_1x1, 64, 192, k=3, s=1, layer_name="conv2")
        lrn2 = tf.nn.lrn(c2, depth_radius=2, bias=1.0, alpha=1e-3, beta=0.75)
        mp2 = max_pool(lrn2, k=3, s=2)

    with tf.variable_scope("l1"):
        l1_1x1 = conv_layer(mp2, 192, 64, k=1, s=1, layer_name="l1_1x1")

        l1_3x3_pre = conv_layer(mp2, 192, 96, k=1, s=1, layer_name="l1_3x3_pre")
        l1_3x3 = conv_layer(l1_3x3_pre, 96, 128, k=3, s=1, layer_name="l1_3x3")

        l1_5x5_pre = conv_layer(mp2, 192, 16, k=1, s=1, layer_name="l1_5x5_pre")
        l1_5x5 = conv_layer(l1_5x5_pre, 16, 32, k=5, s=1, layer_name="l1_5x5")

        l1_mp = max_pool(mp2, k=3, s=1)
        l1_proj = conv_layer(l1_mp, 192, 32, k=1, s=1, layer_name="l1_proj")

        l1 = tf.concat(3, [l1_1x1, l1_3x3, l1_5x5, l1_proj])
        # out here should be 18 x 18 x 256

    with tf.variable_scope("l2"):
        l2_1x1 = conv_layer(l1, 256, 128, k=1, s=1, layer_name="l2_1x1")

        l2_3x3_pre = conv_layer(l1, 256, 128, k=1, s=1, layer_name="l2_3x3_pre")
        l2_3x3 = conv_layer(l2_3x3_pre, 128, 192, k=3, s=1, layer_name="l2_3x3")

        l2_5x5_pre = conv_layer(l1, 256, 32, k=1, s=1, layer_name="l2_5x5_pre")
        l2_5x5 = conv_layer(l2_5x5_pre, 32, 96, k=5, s=1, layer_name="l2_5x5")

        l2_mp = max_pool(l1, k=3, s=1)
        l2_proj = conv_layer(l2_mp, 256, 64, k=1, s=1, layer_name="l2_proj")

        l2_concat = tf.concat(3, [l2_1x1, l2_3x3, l2_5x5, l2_proj])
        # output here is 18 x 18 x 480
        l2 = max_pool(l2_concat, k=3, s=2)
        # output here is 9 x 9 x 480

    with tf.variable_scope("l3"):
        l3_1x1 = conv_layer(l2, 480, 192, k=1, s=1, layer_name="l3_1x1")

        l3_3x3_pre = conv_layer(l2, 480, 96, k=1, s=1, layer_name="l3_3x3_pre")
        l3_3x3 = conv_layer(l3_3x3_pre, 96, 208, k=3, s=1, layer_name="l3_3x3")

        l3_5x5_pre = conv_layer(l2, 480, 16, k=1, s=1, layer_name="l3_5x5_pre")
        l3_5x5 = conv_layer(l3_5x5_pre, 16, 48, k=5, s=1, layer_name="l3_5x5")

        l3_mp = max_pool(l2, k=3, s=1)
        l3_proj = conv_layer(l3_mp, 480, 64, k=1, s=1, layer_name="l3_proj")

        l3 = tf.concat(3, [l3_1x1, l3_3x3, l3_5x5, l3_proj])
        #output here is 9 x 9 x 512

    with tf.variable_scope("l4"):
        l4_1x1 = conv_layer(l3, 512, 160, k=1, s=1, layer_name="l4_1x1")

        l4_3x3_pre = conv_layer(l3, 512, 112, k=1, s=1, layer_name="l4_3x3_pre")
        l4_3x3 = conv_layer(l4_3x3_pre, 112, 224, k=3, s=1, layer_name="l4_3x3")

        l4_5x5_pre = conv_layer(l3, 512, 24, k=1, s=1, layer_name="l4_5x5_pre")
        l4_5x5 = conv_layer(l4_5x5_pre, 24, 64, k=5, s=1, layer_name="l4_5x5")

        l4_mp = max_pool(l3, k=3, s=1)
        l4_proj = conv_layer(l4_mp, 512, 64, k=1, s=1, layer_name="l4_proj")

        l4 = tf.concat(3, [l4_1x1, l4_3x3, l4_5x5, l4_proj])
        #output here is 9 x 9 x 512

    with tf.variable_scope("l5"):
        l5_1x1 = conv_layer(l4, 512, 128, k=1, s=1, layer_name="l5_1x1")

        l5_3x3_pre = conv_layer(l4, 512, 128, k=1, s=1, layer_name="l5_3x3_pre")
        l5_3x3 = conv_layer(l5_3x3_pre, 128, 256, k=3, s=1, layer_name="l5_3x3")

        l5_5x5_pre = conv_layer(l4, 512, 24, k=1, s=1, layer_name="l5_5x5_pre")
        l5_5x5 = conv_layer(l5_5x5_pre, 24, 64, k=5, s=1, layer_name="l5_5x5")

        l5_mp = max_pool(l4, k=3, s=1)
        l5_proj = conv_layer(l5_mp, 512, 64, k=1, s=1, layer_name="l5_proj")

        l5 = tf.concat(3, [l5_1x1, l5_3x3, l5_5x5, l5_proj])
        #output here is 9 x 9 x 512

    with tf.variable_scope("lrn"):
        norm = tf.nn.local_response_normalization(l5, depth_radius=2,
                bias=1.0, alpha=1e-3, beta=0.75)
        avg_pool = tf.nn.avg_pool(norm, [1, 9, 9, 1], [1, 1, 1, 1], 'VALID')
        # output here is 1 x 1 x 480
        # use a deconvolution here instead?
        flat = tf.reshape(avg_pool, [-1, 512])

    with tf.variable_scope("fc"):
        fc1 = fc_layer(flat, 512, 256, keep_prob, "fc1")
        fc2 = fc_layer(fc1, 256, 2, 1.0, "fc2", final=True)

    return fc2

