# -*- coding:utf-8 -*-
from tensorflow.python.framework import ops

import tensorflow as tf
# https://github.com/CS2470FinalProject/X-GAN/blob/master/model.py
class FlipGradientBuilder(object):
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        grad_name = "FlipGradient%d" % self.num_calls
        @ops.RegisterGradient(grad_name)
        def _flip_gradients(op, grad):
            return [tf.negative(grad) * l]
        
        g = tf.compat.v1.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
            y = tf.identity(x)
            
        self.num_calls += 1
        return y
    
flip_gradient = FlipGradientBuilder()

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def conv2d(x, filters, kernel_size=4, strides=2, padding="same"):
    return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                  padding=padding)(x)

def deconv2d(x, filters, kernel_size=4, strides=2):
    return tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides,
                                           padding="same")(x)

def encoder(input_shape=(64, 64, 3), filters=32):
    # [-1, 64, 64, 3]
    h = inputs = tf.keras.Input(input_shape)

    ################## encoder ##################
    h = InstanceNormalization()(conv2d(h, filters=filters))
    h = tf.keras.layers.LeakyReLU()(h)
    h = InstanceNormalization()(conv2d(h, filters=filters*2))
    h = tf.keras.layers.LeakyReLU()(h)
    #############################################

    ################## encoder_sharing ##################
    h = InstanceNormalization()(conv2d(h, filters=filters*4))
    h = tf.keras.layers.LeakyReLU()(h)
    h = InstanceNormalization()(conv2d(h, filters=filters*8))
    h = tf.keras.layers.LeakyReLU()(h)
    h = conv2d(h, filters=1024, kernel_size=4, strides=1, padding="valid")
    h = tf.keras.layers.LeakyReLU()(h)
    h = conv2d(h, filters=1024, kernel_size=1, strides=1, padding="valid")
    h = tf.keras.layers.LeakyReLU()(h)
    #####################################################
    
    return tf.keras.Model(inputs=inputs, outputs=h)

def decoder(input_shape=(1, 1, 1024), filters=64):
    # [-1, 1, 1, 1024]
    h = inputs = tf.keras.Input(input_shape)

    h = InstanceNormalization()(deconv2d(h, filters=filters*8, strides=4))
    h = tf.keras.layers.ReLU()(h)
    h = InstanceNormalization()(deconv2d(h, filters=filters*4))
    h = tf.keras.layers.ReLU()(h)

    h = InstanceNormalization()(deconv2d(h, filters=filters*2))
    h = tf.keras.layers.ReLU()(h)
    h = InstanceNormalization()(deconv2d(h, filters=filters))
    h = tf.keras.layers.ReLU()(h)
    h = deconv2d(h, filters=3)
    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def cdann(input_shape=(1, 1, 1024)):
    # [-1, 1, 1, 1024]
    h = inputs = tf.keras.Input(input_shape)

    fg = flip_gradient(h)
    c1 = conv2d(fg, filters=1024, kernel_size=1, strides=1, padding="valid")
    c1 = tf.keras.layers.LeakyReLU()(c1)
    # c1 is (batch_size x 1 x 1 x 1024)
    c2 = conv2d(fg, filters=1024, kernel_size=1, strides=1, padding="valid")
    c2 = tf.keras.layers.LeakyReLU()(c2)
    # c2 is (batch_size x 1 x 1 x 1024)
    c3 = conv2d(fg, filters=1024, kernel_size=1, strides=1, padding="valid")
    c3 = tf.keras.layers.LeakyReLU()(c3)
    # c3 is (batch_size x 1 x 1 x 1024)
    c4 = conv2d(fg, filters=1, kernel_size=1, strides=1, padding="valid")
    # c4 is (batch_size x 1 x 1 x 1)

    return tf.keras.Model(inputs=inputs, outputs=c4)

def XGAN(input_shape=(64, 64, 3)):  # 모델을 독립적으로 만들어야함
    # 모델을 각각 나누어야하나? 그러면 gradient update도 각각 해주어야함!!! 코드는 길어지겠지만 이방법이 좋아보임
    # encoder와 decoder만 구성!?!?!?!?
    A = A_inputs = tf.keras.Input(input_shape)
    B = B_inputs = tf.keras.Input(input_shape)

    embedding_A = encoder(A)
    embedding_B = encoder(B)

    # A->encoderA->decoderA
    # B->encoderB->decoderB
    reconstruct_A = decoder(embedding_A)
    reconstruct_B = decoder(embedding_B)

    # Cdann output
    cdann_A = cdann(embedding_A)
    cdann_B = cdann(embedding_B)

    # Generator output
    # B->encoderB->decoderA
    # A->encoderA->decoderB
    fake_A = decoder(embedding_B)
    fake_B = decoder(embedding_A)

    embedding_fake_A = encoder(fake_A)
    embedding_fake_B = encoder(fake_B)


    return tf.keras.Model(inputs=[A_inputs, B_inputs], outputs=[reconstruct_A,
                                                                reconstruct_B,
                                                                cdann_A,
                                                                cdann_B,
                                                                embedding_A,
                                                                embedding_B,
                                                                embedding_fake_A,
                                                                embedding_fake_B,
                                                                fake_A,
                                                                fake_B])


def discriminator(input_shape=(64, 64, 3), filters=16):

    h = inputs = tf.keras.Input(input_shape)

    h = conv2d(h, filters=filters)
    h = tf.keras.layers.LeakyReLU()(h)

    h = conv2d(h, filters=filters*2)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = conv2d(h, filters=filters*2)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = conv2d(h, filters=filters*2)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)

    h = conv2d(h, filters=1, kernel_size=4, strides=1, padding="valid")

    return tf.keras.Model(inputs=inputs, outputs=h)
