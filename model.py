import tensorflow as tf
import numpy as np
import math

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

def conv2d(x, W, s, padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=padding)

def max_pool(x, k, s, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1],
            padding=padding)

def conv_layer(input_tensor, input_dim, output_dim, k, s, layer_name, padding='SAME'):
    with tf.variable_scope(layer_name):
        if np.shape(k) == ():
            weights = weight_var("weights", [k, k, input_dim, output_dim], wd=3.0e-4)
        else:
            weights = weight_var("weights", [k[0], k[1], input_dim, output_dim], wd=3.0e-4)

        bias = bias_var("bias", [output_dim])

        variable_summaries(weights, layer_name + '/weights')
        variable_summaries(bias, layer_name + '/biases')

        activations = conv2d(input_tensor, weights, s, padding) + bias
        relu = tf.nn.relu(activations, 'relu')

        tf.histogram_summary(layer_name + '/activations', activations)
        tf.histogram_summary(layer_name + '/activations_relu', relu)

    return relu

def fc_layer(input_tensor, input_dim, output_dim, keep_prob, layer_name, final=False):
    with tf.variable_scope(layer_name):
        weights = weight_var("weights", [input_dim, output_dim], wd=1.0e-3)
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

def deconv_layer(input_tensor, input_dim, output_shape, k, s, layer_name):
    output_dim = output_shape[3]
    with tf.variable_scope(layer_name):
        weights = get_deconv_filter([k[0], k[1], output_dim, input_dim], layer_name + "/deconv_weights")
        return tf.nn.conv2d_transpose(input_tensor, weights, output_shape,
                [1, s, s, 1], padding='VALID')

def get_deconv_filter(f_shape, name):
    width = f_shape[0]
    heigh = f_shape[1]

    f = math.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])

    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value

    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)

    return tf.get_variable(name=name, initializer=init, shape=weights.shape)


def reduce_block(input_tensor, i, name):
    """
    Given an input of size [batch_size, height, width, channels]
    returns an output of size [batch_size, (height - 1)/2, (width - 1)/2, channels + 256]
    """
    #in_shape = tf.shape(input_tensor)
    #i = in_shape[3]

    with tf.variable_scope("reduce/" + name):
        mp = max_pool(input_tensor, k=3, s=2, padding='VALID')
        c1 = conv_layer(input_tensor, i, 192, k=3, s=2, layer_name="c1" + name,
                padding="VALID")
        c2_1x1_pre = conv_layer(input_tensor, i, 96, k=1, s=1, layer_name="c2_1x1_pre" + name)
        c2_1 = conv_layer(c2_1x1_pre, 96, 96, k=3, s=1, layer_name="c2_1" + name)
        c2_2 = conv_layer(c2_1, 96, 64, k=3, s=2, layer_name="c2_2" + name, padding="VALID")

        concat = tf.concat(3, [mp, c1, c2_2])
        return concat

def upsample_block(input_tensor, inshape):
    """
    takes [batch_size, w, h, c] -> [batch_size, 2*w + 1, 2*h + 1, c - diff]
    """
    i = inshape[3]
    o = inshape[3] - 256

    with tf.variable_scope("upsample"):
        return deconv_layer(input_tensor, i, [inshape[0], 2*inshape[1] + 1, 2*inshape[2] + 1, o],
                k=[3,3], s=2, layer_name="deconv")

def down_block(input_tensor, i, j, name):
    in_shape = tf.shape(input_tensor)
    #i = in_shape[3]
    #j = in_shape[3] - diff

    with tf.variable_scope('down_block/' + name):
        c1_1x1_pre = conv_layer(input_tensor, i, j,
            k=1, s=1, layer_name="c1_1x1_pre_db" + name)
        c2_1x1_pre = conv_layer(input_tensor, i, j,
            k=1, s=1, layer_name="c2_1x1_pre_db" + name)
        c3_1x1_pre = conv_layer(input_tensor, i, j,
            k=1, s=1, layer_name="c3_1x1_pre_db" + name)

        c2 = conv_layer(c2_1x1_pre, j, j, k=3, s=1, layer_name="c2" + name)
        c3_1 = conv_layer(c3_1x1_pre, j, j, k=3, s=1, layer_name="c3_1" + name)
        c3_2 = conv_layer(c3_1, j, j, k=3, s=1, layer_name="c3_2" + name)

        concat = tf.concat(3, [c1_1x1_pre, c2, c3_2])
        c4 = conv_layer(concat, 3*j, i, k=1, s=1, layer_name="c4" + name)
        res = tf.add(input_tensor, c4)
        out = tf.nn.relu(res)

        return out, reduce_block(out, i, name)

def up_block(input_tensor, aux_tensor, in_shape, j, q, name):
    i = in_shape[3]
    #j = in_shape[3] - diff

    with tf.variable_scope('up_block/' + name):
        up = upsample_block(input_tensor, in_shape)
        concat_pre = tf.concat(3, [aux_tensor, up])
        # q is the no channels in concat_pre

        concat = conv_layer(concat_pre, q, i, k=3, s=1,
                layer_name="concat_post" + name)

        c1_1x1_pre = conv_layer(concat, i, j,
            k=1, s=1, layer_name="c1_1x1_pre_ub" + name)
        c2_1x1_pre = conv_layer(concat, i, j,
            k=1, s=1, layer_name="c2_1x1_pre_ub" + name)
        c3_1x1_pre = conv_layer(concat, i, j,
            k=1, s=1, layer_name="c3_1x1_pre_ub" + name)

        c2 = conv_layer(c2_1x1_pre, j, j, k=3, s=1, layer_name="c2" + name)
        c3_1 = conv_layer(c3_1x1_pre, j, j, k=3, s=1, layer_name="c3_1" + name)
        c3_2 = conv_layer(c3_1, j, j, k=3, s=1, layer_name="c3_2" + name)

        conv_concat = tf.concat(3, [c1_1x1_pre, c2, c3_2])
        c4 = conv_layer(conv_concat, 3*j, i, k=1, s=1, layer_name="c4" + name)
        res = tf.add(concat, c4)

        out = tf.nn.relu(res)
        # need to add this in to actually reduce the number of channels
        out_post = conv_layer(out, i, j, k=1, s=1, layer_name="out_post" + name)

        return out_post

def base(input_tensor, i):
    #in_shape = tf.shape(input_tensor)
    #i = in_shape[3]

    with tf.variable_scope("base"):
        avg = tf.nn.avg_pool(input_tensor, [1, 3, 3, 1], [1, 1, 1, 1], padding='SAME')
        avg_1x1_post = conv_layer(avg, i, 128, k=1, s=1, layer_name="avg_1x1_post")

        c_1_1x1 = conv_layer(input_tensor, i, 384, k=1, s=1, layer_name="c_1_1x1")

        c_2_1x1_pre = conv_layer(input_tensor, i, 192, k=1, s=1, layer_name="c_2_1x1_pre")
        c_2_1x7 = conv_layer(c_2_1x1_pre, 192, 224, k=[1,7], s=1, layer_name="c_2_1x7")
        c_2_7x1 = conv_layer(c_2_1x7, 224, 256, k=[7,1], s=1, layer_name="c_2_7x1")

        c_3_1x1_pre = conv_layer(input_tensor, i, 192, k=1, s=1, layer_name="c_3_1x1_pre")
        c_3_1x7_1 = conv_layer(c_3_1x1_pre, 192, 224, k=[1,7], s=1, layer_name="c_3_1x7_1")
        c_3_7x1_1 = conv_layer(c_3_1x7_1, 224, 224, k=[7,1], s=1, layer_name="c_3_7x1_1")
        c_3_1x7_2 = conv_layer(c_3_7x1_1, 224, 224, k=[1,7], s=1, layer_name="c_3_1x7_2")
        c_3_7x1_2 = conv_layer(c_3_1x7_2, 224, 256, k=[7,1], s=1, layer_name="c_3_7x1_2")

        concat = tf.concat(3, [avg_1x1_post, c_1_1x1, c_2_7x1, c_3_7x1_2])
        return concat

def stem(input_tensor):
    """
    Input has shape 420 x 580 x 1

    Output has shape 51 x 71 x 256
    """
    with tf.variable_scope("stem"):
        padded = tf.pad(input_tensor, [[0,0], [0,3], [0,3], [0,0]], "CONSTANT")
        c1 = conv_layer(padded, 1, 32, k=3, s=2, layer_name="c1-3x3-2-V",
                padding="VALID")
        c2 = conv_layer(c1, 32, 64, k=3, s=1, layer_name="c2-3x3-1-S")
        mp1 = max_pool(c2, k=3, s=2, padding="VALID")
        c3 = conv_layer(c2, 64, 96, k=3, s=2, layer_name="c3-3x3-2-V",
                padding="VALID")
        concat = tf.concat(3, [mp1, c3])
        c4 = conv_layer(concat, 160, 196, k=3, s=1, layer_name="c4-3x3-1-V",
                padding="VALID")
        c5 = conv_layer(c4, 196, 256, k=3, s=2, layer_name="c5-3x3-2-V",
                padding="VALID")

        return c5

def inference(images, batch_size):
    with tf.variable_scope("stem"):
        init = stem(images)
    with tf.variable_scope("l1"):
        skip1, l1 = down_block(init, 256, 128, "l1")
    with tf.variable_scope("l2"):
        skip2, l2 = down_block(l1, 512, 386, "l2")
    with tf.variable_scope("base"):
        b = base(l2, 768)
    with tf.variable_scope("u1"):
        u1 = up_block(b, skip2, [batch_size, 12, 17, 1024], 512, 1280, "u1")
    with tf.variable_scope("u2"):
        u2 = up_block(u1, skip1, [batch_size, 25, 35, 512], 256, 512, "u2")
    with tf.variable_scope("final_deconv"):
        # output here is batch_size x 51 x 71 x 256
        final = deconv_layer(u2, 256, [batch_size, 420, 580, 2], k=[20,20], s=8,
                layer_name="final_deconv")

    return final

