# -*- coding:utf-8 -*-
from random import shuffle
from XGAN_model import *

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import easydict

FLAGS = easydict.EasyDict({"img_size": 64,
                           
                           "batch_size": 16,
                           
                           "lr": 0.0001,

                           "epochs": 100,
                           
                           "A_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/female_16_39_train.txt",
                           
                           "A_img_path": "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/",
                           
                           "B_txt_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/male_40_63_train.txt",
                           
                           "B_img_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/male_40_63/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "",
                           
                           "save_checkpoint": "",
                           
                           "sample_images": "C:/Users/Yuhwan/Downloads/dd"})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def _func(A_filename, B_filename):

    h = tf.random.uniform([1], 1e-2, 5)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 5)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)
    
    A_image_string = tf.io.read_file(A_filename)
    A_decode_image = tf.image.decode_jpeg(A_image_string, channels=3)
    A_decode_image = tf.image.resize(A_decode_image, [FLAGS.img_size + 5, FLAGS.img_size + 5])
    A_decode_image = A_decode_image[h:h+FLAGS.img_size, w:w+FLAGS.img_size, :]
    A_decode_image = tf.image.convert_image_dtype(A_decode_image, tf.float32) / 127.5 - 1.

    B_image_string = tf.io.read_file(B_filename)
    B_decode_image = tf.image.decode_jpeg(B_image_string, channels=3)
    B_decode_image = tf.image.resize(B_decode_image, [FLAGS.img_size + 5, FLAGS.img_size + 5])
    B_decode_image = B_decode_image[h:h+FLAGS.img_size, w:w+FLAGS.img_size, :]
    B_decode_image = tf.image.convert_image_dtype(B_decode_image, tf.float32) / 127.5 - 1.

    if tf.random.uniform(()) > 0.5:
        A_decode_image = tf.image.flip_left_right(A_decode_image)
        B_decode_image = tf.image.flip_left_right(B_decode_image)

    return A_decode_image, B_decode_image

def euc_criterion(in_, target):
    return tf.keras.losses.MeanSquaredError()(in_, target)
def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))
def mae_criterion(in_, target):
    return tf.reduce_mean((in_ - target)**2)
def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

def cal_loss(en_domain_A,
             en_domain_B, 
             de_domain_A,
             de_domain_B,
             Cdann, A_de_model, B_de_model, A_images, B_images):

    with tf.GradientTape(persistent=True) as g_tape, tf.GradientTape(persistent=True) as d_tape:

        # Encoder output
        embedding_A = en_domain_A(A_images, True) # 또 잘못짯나?
        embedding_B = en_domain_B(B_images, True)

        # Reconstruction output
        # A->encoderA->decoderA
        # B->encoderB->decoderB
        reconstruct_A = de_domain_A(embedding_A, True)
        reconstruct_B = de_domain_B(embedding_B, True)

        # Cdann output
        cdann_A = Cdann(embedding_A, True)
        cdann_B = Cdann(embedding_B, True)

        # Generator output
        # B->encoderB->decoderA
        # A->encoderA->decoderB
        fake_A = de_domain_A(embedding_B, True)
        fake_B = de_domain_B(embedding_A, True)

        # Fake image encoder output
        embedding_fake_A = en_domain_A(fake_A, True)
        embedding_fake_B = en_domain_B(fake_B, True)


        # Discrim output
        discriminate_real_A = A_de_model(A_images, True)
        discriminate_real_B = B_de_model(B_images, True)
        discriminate_fake_A = A_de_model(fake_A, True)
        discriminate_fake_B = B_de_model(fake_B, True)

        # Recon loss
        rec_loss_A = euc_criterion(A_images, reconstruct_A)
        rec_loss_B = euc_criterion(B_images, reconstruct_B)
        rec_loss = rec_loss_A + rec_loss_B

        # domain-advar loss
        dann_loss = sce_criterion(cdann_A, tf.zeros_like(cdann_A)) + sce_criterion(cdann_B, tf.ones_like(cdann_B))

        # semantic consistency loss
        sem_loss_A = abs_criterion(embedding_A, embedding_fake_B)
        sem_loss_B = abs_criterion(embedding_B, embedding_fake_A)
        sem_loss = sem_loss_A + sem_loss_B

        # GAN loss-generator
        gen_gan_loss_A = mae_criterion(discriminate_fake_A, tf.ones_like(discriminate_fake_A))
        gen_gan_loss_B = mae_criterion(discriminate_fake_B, tf.ones_like(discriminate_fake_B))
        gen_gan_loss = gen_gan_loss_A + gen_gan_loss_B

        # Total loss
        gen_loss = rec_loss + dann_loss + sem_loss + gen_gan_loss

        # Gan loss-discriminator
        dis_gan_loss_real_A = mae_criterion(discriminate_real_A, tf.ones_like(discriminate_real_A))
        dis_gan_loss_fake_A = mae_criterion(discriminate_fake_A, tf.zeros_like(discriminate_fake_A))
        dis_gan_loss_A = (dis_gan_loss_real_A + dis_gan_loss_fake_A) / 2
        dis_gan_loss_real_B = mae_criterion(discriminate_real_B, tf.ones_like(discriminate_real_B))
        dis_gan_loss_fake_B = mae_criterion(discriminate_fake_B, tf.zeros_like(discriminate_fake_B))
        dis_gan_loss_B = (dis_gan_loss_real_B + dis_gan_loss_fake_B) / 2
        dis_gan_loss = dis_gan_loss_A + dis_gan_loss_B

        # Total loss
        dis_loss = dis_gan_loss

    g_grads1 = g_tape.gradient(gen_loss, en_domain_A.trainable_variables + en_domain_B.trainable_variables)
    g_grads2 = g_tape.gradient(gen_loss, de_domain_A.trainable_variables + de_domain_B.trainable_variables)
    g_grads3 = g_tape.gradient(gen_loss, Cdann.trainable_variables)

    d_grads = d_tape.gradient(dis_loss, A_de_model.trainable_variables + B_de_model.trainable_variables)

    g_optim.apply_gradients(zip(g_grads1, en_domain_A.trainable_variables + en_domain_B.trainable_variables))
    g_optim.apply_gradients(zip(g_grads2, de_domain_A.trainable_variables + de_domain_B.trainable_variables))
    g_optim.apply_gradients(zip(g_grads3, Cdann.trainable_variables))
    d_optim.apply_gradients(zip(d_grads, A_de_model.trainable_variables + B_de_model.trainable_variables))


    return gen_loss, dis_loss

def main():

    en_domain_A = encoder(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    en_domain_B = encoder(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    de_domain_A = decoder(input_shape=(1, 1, 1024))
    de_domain_B = decoder(input_shape=(1, 1, 1024))

    Cdann = cdann(input_shape=(1, 1, 1024))

    A_de_model = discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B_de_model = discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    en_domain_A.summary()
    de_domain_A.summary()
    Cdann.summary()
    A_de_model.summary()
    B_de_model.summary()


    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(A_ge_model=A_ge_model, B_ge_model=B_ge_model, A_de_model=A_de_model, B_de_model=B_de_model,
                                   g_optim=g_optim, d_optim=d_optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    A_input = np.loadtxt(FLAGS.A_txt_path, dtype="<U200", skiprows=0, usecols=0)
    A_input = [FLAGS.A_img_path + data for data in A_input]

    B_input = np.loadtxt(FLAGS.B_txt_path, dtype="<U200", skiprows=0, usecols=0)
    B_input = [FLAGS.B_img_path + data for data in B_input]

    count = 0
    for epoch in range(FLAGS.epochs):

        shuffle(A_input)
        shuffle(B_input)

        A_input, B_input = np.array(A_input), np.array(B_input)
        
        tr_gener = tf.data.Dataset.from_tensor_slices((A_input, B_input))
        tr_gener = tr_gener.shuffle(len(A_input))
        tr_gener = tr_gener.map(_func)
        tr_gener = tr_gener.batch(FLAGS.batch_size)
        tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)
        
        tr_iter = iter(tr_gener)
        tr_idx = len(A_input) // FLAGS.batch_size
        for step in range(tr_idx):
            A_images, B_images = next(tr_iter)
            a = 0

            g_loss, d_loss = cal_loss(en_domain_A,
                                      en_domain_B, 
                                      de_domain_A,
                                      de_domain_B,
                                      Cdann,
                                      A_de_model, 
                                      B_de_model, 
                                      A_images, B_images)

            if count % 100 == 0:
                print("Epochs: {} g_loss = {} d_loss = {} [{}/{}]".format(epoch, g_loss, d_loss, step + 1, tr_idx))

                embedding_A = en_domain_A(A_images, False)
                embedding_B = en_domain_B(B_images, False)

                fake_A = de_domain_A(embedding_B, False)
                fake_B = de_domain_B(embedding_A, False)
                plt.imsave(FLAGS.sample_images + "/" + "{}_1_A_real.png".format(count), A_images[0].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_2_A_real.png".format(count), A_images[1].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_3_A_real.png".format(count), A_images[2].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_1_B_real.png".format(count), B_images[0].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_2_B_real.png".format(count), B_images[1].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_3_B_real.png".format(count), B_images[2].numpy() * 0.5 + 0.5)

                plt.imsave(FLAGS.sample_images + "/" + "{}_1_A_fake.png".format(count), fake_A[0].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_2_A_fake.png".format(count), fake_A[1].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_3_A_fake.png".format(count), fake_A[2].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_1_B_fake.png".format(count), fake_B[0].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_2_B_fake.png".format(count), fake_B[1].numpy() * 0.5 + 0.5)
                plt.imsave(FLAGS.sample_images + "/" + "{}_3_B_fake.png".format(count), fake_B[2].numpy() * 0.5 + 0.5)



            count += 1

if __name__ == "__main__":
    main()
