import os, sys, shutil, gzip, argparse, math
import gzip
from tqdm import tqdm
import numpy as np
from typing import Union
from scipy.special import softmax
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import tensorflow_addons as tfa
from scipy.spatial.distance import squareform
import datatable as dt
import json
import os
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D, Conv2D, PReLU,LayerNormalization, Flatten, UpSampling2D, LeakyReLU, Dense, Input, add, Lambda
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, BatchNormalization, Activation, MaxPool2D, Concatenate, Add


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def pprint(text):
    print(f"{bcolors.OKGREEN}{text}{bcolors.ENDC}")

def find_next_multiple_of_m(number, m):
  ceiling_value = math.ceil(number)
  if ceiling_value % m == 0:
      return ceiling_value
  next_multiple_of_m = ceiling_value + (m - (ceiling_value % m))

  return int(next_multiple_of_m)

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# logging.basicConfig(level=logging.WARNING)
pprint("Tensorflow version " + tf.__version__)


keras.saving.get_custom_objects().clear()

## Custom Layers
@keras.saving.register_keras_serializable(package="MyLayers")
class MultiHeadAttnBlock2D(layers.Layer):
    def __init__(self, inter_channel, num_heads,
                dropout_rate=0.1):
        super(MultiHeadAttnBlock2D, self).__init__()
        regularizers = tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-3)
        channels_per_head = inter_channel // num_heads
        self.num_heads = num_heads
        self.theta_x_heads = [layers.Dense(channels_per_head, activation=tf.nn.gelu,
                      kernel_regularizer=regularizers) for _ in range(num_heads)]
        self.phi_g_heads = [layers.Dense(channels_per_head, activation=tf.nn.softmax,
                      kernel_regularizer=regularizers) for _ in range(num_heads)]
        self.rate_head = layers.Activation('sigmoid')
        self.dropout = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        super(MultiHeadAttnBlock2D, self).build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
            "num_heads": self.num_heads,
            "theta_x_heads": self.theta_x_heads,
            "theta_x_heads": self.theta_x_heads,
            "phi_g_heads": self.phi_g_heads,
            "rate_head": self.rate_head,
            "dropout": self.dropout,
            }
        )
        return config


    def call(self, inputs):
        x, g = inputs
        head_outputs = []

        for i in range(self.num_heads):
            # Extract the slice for this head
            theta_x_head = self.theta_x_heads[i](x)
            phi_g_head = self.phi_g_heads[i](g)

            # Apply the attention mechanism per head
            f_head = layers.Multiply()([theta_x_head, phi_g_head])
            head_outputs.append(f_head)

        # Concatenate the outputs from all heads
        multi_head_output = layers.Concatenate()(head_outputs)
        multi_head_output = self.dropout(multi_head_output)
        rate_head = self.rate_head(multi_head_output)
        output = layers.Multiply()([rate_head, x])

        return output

    
@keras.saving.register_keras_serializable(package="MyLayers")
class TransformerBlock2D(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, activation=tf.nn.gelu,
                dropout_rate=0.0):
        super(TransformerBlock2D, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        self.att0 = MultiHeadAttnBlock2D(num_heads=self.num_heads,
                                            inter_channel=self.embed_dim,
                                        dropout_rate=dropout_rate)
        regularizers = tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-3)
        self.ffn = tf.keras.Sequential(
            [
            layers.Dense(self.ff_dim, activation=self.activation,
                        kernel_regularizer=regularizers
                        ),
            layers.Dense(self.embed_dim,
                        activation=self.activation,
                        kernel_regularizer=regularizers
                        ), ]
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "att0": self.att0,
                "ffn": self.ffn,
                "layer_norm1": self.layer_norm1,
                "layer_norm2": self.layer_norm2,
            }
        )
        return config
    
    def call(self, inputs):
        x = self.layer_norm1(inputs)
        attn_output = self.att0([x, x])
        ffn_output = self.ffn(x)
        return self.layer_norm2(attn_output + ffn_output)

@keras.saving.register_keras_serializable(package="MyLayers", name="conv_block")
def conv_block(inputs, out_ch, rate=1):
    regularizers = tf.keras.regularizers.L1L2(l1=1e-4, l2=1e-3)
    x = Conv2D(out_ch, 3, padding="same", dilation_rate=rate,
                      kernel_regularizer=regularizers)(inputs)
    # x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

@keras.saving.register_keras_serializable(package="MyLayers", name="RSU_L")
def RSU_L(inputs, out_ch, int_ch, num_layers, rate=2):
    """ Initial Conv """
    x = conv_block(inputs, out_ch)
    init_feats = x

    """ Encoder """
    skip = []
    x = conv_block(x, int_ch)
    skip.append(x)

    for i in range(num_layers-2):
        # x = MaxPool2D((2, 2))(x)
        x = conv_block(x, int_ch)
        skip.append(x)

    """ Bridge """
    x = conv_block(x, int_ch, rate=rate)

    """ Decoder """
    skip.reverse()

    x = Concatenate()([x, skip[0]])
    x = conv_block(x, int_ch)

    for i in range(num_layers-3):
        # x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = Concatenate()([x, skip[i+1]])
        x = conv_block(x, int_ch)

    # x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Concatenate()([x, skip[-1]])
    x = conv_block(x, out_ch)

    """ Add """
    x = Add()([x, init_feats])
    return x

@keras.saving.register_keras_serializable(package="MyLayers", name="RSU_4F")
def RSU_4F(inputs, out_ch, int_ch):
    """ Initial Conv """
    x0 = conv_block(inputs, out_ch, rate=1)

    """ Encoder """
    x1 = conv_block(x0, int_ch, rate=1)
    x2 = conv_block(x1, int_ch, rate=2)
    x2 = layers.Dropout(0.3)(x2)
    x3 = conv_block(x2, int_ch, rate=4)

    """ Bridge """
    x4 = conv_block(x3, int_ch, rate=8)

    """ Decoder """
    x = Concatenate()([x4, x3])
    x = conv_block(x, int_ch, rate=4)
    
    x = Concatenate()([x, x2])
    x = conv_block(x, int_ch, rate=2)
    x = layers.Dropout(0.3)(x)
    x = Concatenate()([x, x1])
    x = conv_block(x, out_ch, rate=1)

    """ Addition """
    x = Add()([x, x0])
    return x

@keras.saving.register_keras_serializable(package="MyLayers", name="u2net")
def u2net(input_shape, out_ch, int_ch, n_heads=16):
    """ Input Layer """
    s0 = inputs = Input(input_shape)
    num_channels = input_shape[-1]

    """ Encoder """
    s1 = RSU_L(s0, out_ch[0], int_ch[0], 3)

    s5 = RSU_4F(s1, out_ch[1], int_ch[1])

    """ Bridge """
    b1 = TransformerBlock2D(out_ch[2], n_heads, out_ch[2]//2)(s5)

    """ Decoder """
    d1 = Concatenate()([b1, s5])
    d1 = RSU_4F(d1, out_ch[-2], int_ch[-2])

    d5 = Concatenate()([d1, s1])
    d5 = RSU_L(d5, out_ch[-1], int_ch[-1], 3)

    """ Side Outputs """
    y1 = Conv2D(num_channels, 3, padding="same")(d5) + inputs[..., 0:1]

    y4 = Conv2D(num_channels, 3, padding="same")(d1) + inputs[..., 0:1]

    y6 = Conv2D(num_channels, 3, padding="same")(b1) + inputs[..., 0:1]
    y0 = Concatenate()([y1, y4, y6])
    y0 = Conv2D(num_channels, 3, padding="same")(y0)

    y0 = Activation("relu")(y0)
    y1 = Activation("relu")(y1)
    y4 = Activation("relu")(y4)
    y6 = Activation("relu")(y6)

    model = tf.keras.models.Model(inputs, outputs=[y0, y1, y4, y6])
    return model

@keras.saving.register_keras_serializable(package="MyLayers", name="create_gen")
def create_gen(input_shape, n_heads):
    out_ch = [128, 256, 256, 256, 256, 256, 256, 256, 256, 256, 128]
    int_ch = [64, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64]
    model = u2net(input_shape, out_ch, int_ch, n_heads=n_heads)
    return model


@keras.saving.register_keras_serializable(package="MyLayers", name="create_gen_lite")
def create_gen_lite(input_shape, n_heads):
    out_ch = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
    int_ch = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    model = u2net(input_shape, out_ch, int_ch, n_heads=n_heads)
    return model


## TruHiC Model
@keras.saving.register_keras_serializable(package="MyModels")
class ModelWrapper(tf.keras.Model):
    def __init__(self, input_shape, global_batch_size=16,
                 n_heads=16,
                 symmetry_loss_weight=1e-2, mse_loss_weight=1,
                 insulation_loss_weight=1, gen_gradient_loss_weight=1e-2,
                 insulation_window_radius=5, insulation_deriv_size=5,
                 psnr_loss_weight=1, **kwargs):
        super().__init__(**kwargs)
        self.in_shape = input_shape
        self.symmetry_loss_weight = symmetry_loss_weight
        self.mse_loss_weight = mse_loss_weight
        self.generator = create_gen(input_shape, n_heads)
        self.gen_gradient_loss_weight = gen_gradient_loss_weight
        self.insulation_loss_weight = insulation_loss_weight
        self.psnr_loss_weight = psnr_loss_weight
        self.psnr_loss_tracker = tf.keras.metrics.Mean(name="psnr_loss")
        self.global_batch_size = global_batch_size

        self.di_loss_weight = 1e-2

        self.num_outputs = 4
        self.mse_fn = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE,
            name='mean_squared_error'
        )
        self.bce_fn = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        # self.ins_scorer_model = InsulationScoreModule(window_radius=insulation_window_radius,
        #                                       deriv_size=insulation_deriv_size)
        self.mse_loss_tracker = tf.keras.metrics.Mean(name="mse_loss")
        self.gen_symm_tracker = tf.keras.metrics.Mean(name="gen_symm_loss")
        self.gen_grad_tracker = tf.keras.metrics.Mean(name="gen_gradient_loss")
        self.di_loss_tracker = tf.keras.metrics.Mean(name="di_loss")
        # self.gen_insulation_tracker = tf.keras.metrics.Mean(name="gen_ins_loss")

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                # "gen_insulation_tracker": self.gen_insulation_tracker,
                "gen_grad_tracker": self.gen_grad_tracker,
                "gen_symm_tracker": self.gen_symm_tracker,
                "mse_loss_tracker": self.mse_loss_tracker,
                "di_loss_tracker": self.di_loss_tracker,
                "di_loss_weight": self.di_loss_weight,
                "bce_fn": self.bce_fn,
                "mse_fn": self.mse_fn,
                "num_outputs": self.num_outputs,
                "global_batch_size": self.global_batch_size,
                "psnr_loss_tracker": self.psnr_loss_tracker,
                "psnr_loss_weight": self.psnr_loss_weight,
                "insulation_loss_weight": self.insulation_loss_weight,
                "gen_gradient_loss_weight": self.gen_gradient_loss_weight,
                "generator": self.generator,
                "mse_loss_weight": self.mse_loss_weight,
                "symmetry_loss_weight": self.symmetry_loss_weight,
                "in_shape": self.in_shape,
            }
        )
        return config
    
    @property
    def metrics(self):
        return [
            self.mse_loss_tracker,
            self.psnr_loss_tracker,
            self.di_loss_tracker,
            # self.gen_insulation_tracker,
            # self.gen_symm_tracker,
            # self.gen_grad_tracker,
        ]
    
    def di_loss_v3(self, y_true, y_pred):
        width = y_true.shape[-2]
        batch_size = tf.shape(y_true)[0]
        total_loss = tf.zeros(batch_size)

        # Iterate through possible division points from 1 to width-2
        for i in range(1, width - 1):
            A_true = tf.reduce_mean(y_true[:, :, :i, :], axis=[1, 2, 3])
            B_true = tf.reduce_mean(y_true[:, :, i:, :], axis=[1, 2, 3])
            A_pred = tf.reduce_mean(y_pred[:, :, :i, :], axis=[1, 2, 3])
            B_pred = tf.reduce_mean(y_pred[:, :, i:, :], axis=[1, 2, 3])
            

            E_true = (A_true + B_true) / 2
            E_pred = (A_pred + B_pred) / 2

            numerator_true = (B_true - A_true) / tf.where(tf.math.abs(B_true - A_true) > 1e-6, tf.math.abs(B_true - A_true), 1e-6)
            denominator_true = tf.where(E_true > 1e-6, ((A_true - E_true)**2 / E_true) + ((B_true - E_true)**2 / E_true),\
                                        ((A_true - E_true)**2 / 1e-6) + ((B_true - E_true)**2 / 1e-6))
            DI_true = numerator_true * denominator_true

            numerator_pred = (B_pred - A_pred) / tf.where(tf.math.abs(B_pred - A_pred) > 1e-6, tf.math.abs(B_pred - A_pred), 1e-6)
            denominator_pred = tf.where(E_pred > 1e-6, ((A_pred - E_pred)**2 / E_pred) + ((B_pred - E_pred)**2 / E_pred),\
                                        ((A_pred - E_pred)**2 / 1e-6) + ((B_pred - E_pred)**2 / 1e-6))
            DI_pred = numerator_pred * denominator_pred
            DI_diff = DI_true - DI_pred
            # loss = tf.reduce_mean(DI_diff**2, axis=-1)
            total_loss += DI_diff**2

        # Average the total loss over all divisions
        final_loss = total_loss / tf.cast(width - 2, dtype=tf.float32)
        return final_loss

    def di_loss(self, y_true, y_pred):
        upper_band_true = tf.linalg.band_part(y_true, 0, -1)
        upper_band_pred = tf.linalg.band_part(y_pred, 0, -1)

        lower_band_true = tf.linalg.band_part(y_true, -1, 0)
        lower_band_pred = tf.linalg.band_part(y_pred, -1, 0)

        A_true = tf.expand_dims(tf.reduce_mean(upper_band_true, axis=[1, 2, 3]), -1)
        B_true = tf.expand_dims(tf.reduce_mean(lower_band_true, axis=[1, 2, 3]), -1)


        A_pred = tf.expand_dims(tf.reduce_mean(upper_band_pred, axis=[1, 2, 3]), -1)
        B_pred = tf.expand_dims(tf.reduce_mean(lower_band_pred, axis=[1, 2, 3]), -1)

        E_true = (A_true + B_true) / 2
        E_pred = (A_pred + B_pred) / 2

        numerator_true = (B_true - A_true) / tf.where(tf.math.abs(B_true - A_true) > 1e-6, tf.math.abs(B_true - A_true), 1e-6)
        denominator_true = tf.where(E_true > 1e-6, ((A_true - E_true)**2 / E_true) + ((B_true - E_true)**2 / E_true),\
                                     ((A_true - E_true)**2 / 1e-6) + ((B_true - E_true)**2 / 1e-6))
        DI_true = numerator_true * denominator_true

        numerator_pred = (B_pred - A_pred) / tf.where(tf.math.abs(B_pred - A_pred) > 1e-6, tf.math.abs(B_pred - A_pred), 1e-6)
        denominator_pred = tf.where(E_pred > 1e-6, ((A_pred - E_pred)**2 / E_pred) + ((B_pred - E_pred)**2 / E_pred),\
                                    ((A_pred - E_pred)**2 / 1e-6) + ((B_pred - E_pred)**2 / 1e-6))
        DI_pred = numerator_pred * denominator_pred

        DI_diff = DI_true - DI_pred
        # loss = tf.reduce_mean(DI_diff**2, axis=1)
        return DI_diff**2


    # Define the loss functions for the generator.
    def generator_loss1(self, fake_img):
        transposed_img = tf.transpose(fake_img, perm=[0, 2, 1, 3])
        diff = tf.math.abs(fake_img - transposed_img)
        # symmetrical_loss = tf.reduce_mean(diff)
        symmetrical_loss = tf.reduce_mean(diff, axis=[1,2,3])

        return symmetrical_loss

    def image_gradient_penalty(self, y_true, y_pred):
        real_images = tf.cast(y_true, y_pred.dtype)
        dy_r, dx_r = tf.image.image_gradients(real_images)
        dy_f, dx_f = tf.image.image_gradients(y_pred)
        diff_y = tf.math.pow(tf.math.abs(dy_r - dy_f), 2)
        diff_x = tf.math.pow(tf.math.abs(dx_r - dx_f), 2)
        loss = tf.reduce_mean(diff_y, axis=[1,2,3]) + tf.reduce_mean(diff_x, axis=[1,2,3])
        return loss

    def insulation_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        se = tf.math.pow(y_true - y_pred, 2)
        mse = tf.reduce_sum(se, axis=[1,2])
        return mse

    def noiseToSignalLoss(self, y_true, y_pred):
        losses = tf.math.divide(
            tf.math.reduce_sum(
                tf.math.pow(
                    tf.math.abs(
                        tf.math.subtract(
                            y_true,
                            y_pred
                        )
                    ),
                    2
                ),
                axis=[1,2,3]),
            tf.math.reduce_sum(
                tf.math.pow(tf.math.abs(y_true),2), axis=[1,2,3]
            )
        )
        return losses


    def compute_loss(self, loss_object, labels, predictions, model_losses):
        per_example_loss = loss_object(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss,
                                            global_batch_size=self.global_batch_size)
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss

    def compute_self_loss(self, loss_object, labels, model_losses):
        per_example_loss = loss_object(labels)
        loss = tf.nn.compute_average_loss(per_example_loss,
                                        global_batch_size=self.global_batch_size)
        if model_losses:
            loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
        return loss

    def call(self, data, training=False):
        fake_imgs = self.generator(data, training=training)
        return fake_imgs

    def train_step(self, data):
        lr, hr = data
        ## for generator
        with tf.GradientTape() as g_tape1:
            fake_imgs = self(lr, training=True) #Fake images
            #First, train the discriminator on fake images.
            symm_loss = 0
            if self.symmetry_loss_weight > 0:
                for ind in range(self.num_outputs):
                    symm_loss += self.compute_self_loss(self.generator_loss1, fake_imgs[ind], self.generator.losses)/tf.math.exp(tf.cast(ind, tf.float32))
            # Calculate the generator loss using discriminator
            g_gradient_loss = 0
            if self.gen_gradient_loss_weight > 0:
                for ind in range(self.num_outputs):
                    g_gradient_loss += self.compute_loss(self.image_gradient_penalty, hr, fake_imgs[ind], self.generator.losses)/tf.math.exp(tf.cast(ind, tf.float32))
            # Insulation loss
            # ins_loss = 0
            # if self.insulation_loss_weight > 0:
            #     for ind in range(self.num_outputs):
            #         fake_insulation_score = self.ins_scorer_model(fake_imgs[ind], training=False)
            #         real_insulation_score = self.ins_scorer_model(hr, training=False)
            #         ins_loss += self.compute_loss(self.insulation_loss, real_insulation_score, fake_insulation_score, self.generator.losses)/tf.math.exp(tf.cast(ind, tf.float32))

            # MSE loss
            mse_loss = 0
            for ind in range(self.num_outputs):
                mse_loss += self.compute_loss(self.mse_fn, hr, fake_imgs[ind], self.generator.losses)/tf.math.exp(tf.cast(ind, tf.float32))
            # PSNR loss
            psnr_loss = 0
            if self.psnr_loss_weight > 0:
                for ind in range(self.num_outputs):
                    psnr_loss += self.compute_loss(self.noiseToSignalLoss, hr, fake_imgs[ind], self.generator.losses)/tf.math.exp(tf.cast(ind, tf.float32))

            di_loss = 0
            # for ind in range(self.num_outputs):
            #     di_loss += self.compute_loss(self.di_loss_v3, hr, fake_imgs[ind], self.generator.losses)/tf.math.exp(tf.cast(ind, tf.float32))
            #     di_loss += self.compute_loss(self.di_loss, hr, fake_imgs[ind], self.generator.losses)/tf.math.exp(tf.cast(ind, tf.float32))



            # vgg_loss /= len(hr_features_array)
            # Weighted average
            g_total_cost = mse_loss * self.mse_loss_weight+\
                            symm_loss*self.symmetry_loss_weight +\
                            g_gradient_loss*self.gen_gradient_loss_weight +\
                            di_loss*.5 +\
                            psnr_loss*self.psnr_loss_weight# +\
                            # ins_loss*self.insulation_loss_weight 

        # Get the gradients w.r.t the generator loss
        g_gradient = g_tape1.gradient(g_total_cost, self.generator.trainable_variables)
        # Update the weights of the generator using the generaor optimizer
        self.optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables)
        )


        self.mse_loss_tracker.update_state(mse_loss)
        self.psnr_loss_tracker.update_state(psnr_loss)
        self.di_loss_tracker.update_state(di_loss)
        # self.gen_insulation_tracker.update_state(ins_loss)
        self.gen_grad_tracker.update_state(g_gradient_loss)
        self.gen_symm_tracker.update_state(symm_loss) # For Tpu Training

        return {
            # "gen_ins_loss": self.gen_insulation_tracker.result(),
            "psnr_loss": self.psnr_loss_tracker.result(),
            "mse_loss": self.mse_loss_tracker.result(),
            # "gen_symm_loss": self.gen_symm_tracker.result(),
            # "gen_gradient_loss": self.gen_grad_tracker.result(),
            }

    def test_step(self, data):
        lr, hr = data
        ## for generator
        fake_imgs = self(lr, training=False)[0] #Fake images
        #First, train the discriminator on fake images.
        symm_loss = 0
        if self.symmetry_loss_weight > 0:
            symm_loss = self.compute_self_loss(self.generator_loss1, fake_imgs, self.generator.losses)
        # Calculate the generator loss using discriminator
        # g_cost_adversarial = self.generator_loss2(disc_fake_pred)
        g_gradient_loss = 0
        if self.gen_gradient_loss_weight > 0:
            g_gradient_loss = self.compute_loss(self.image_gradient_penalty, hr, fake_imgs, self.generator.losses)
        # Insulation loss
        # ins_loss = 0
        # if self.insulation_loss_weight > 0:
        #     fake_insulation_score = self.ins_scorer_model(fake_imgs, training=False)
        #     real_insulation_score = self.ins_scorer_model(hr, training=False)
        #     ins_loss = self.compute_loss(self.insulation_loss, real_insulation_score, fake_insulation_score, self.generator.losses)

        # MSE loss
        mse_loss = self.compute_loss(self.mse_fn, hr, fake_imgs, self.generator.losses)
        # PSNR loss
        psnr_loss = 0
        if self.psnr_loss_weight > 0:
            psnr_loss = self.compute_loss(self.noiseToSignalLoss, hr, fake_imgs, self.generator.losses)
        di_loss = 0
        # di_loss += self.compute_loss(self.di_loss_v3, hr, fake_imgs, self.generator.losses)
        # di_loss += self.compute_loss(self.di_loss, hr, fake_imgs, self.generator.losses)
        

        self.mse_loss_tracker.update_state(mse_loss)
        self.psnr_loss_tracker.update_state(psnr_loss)
        self.di_loss_tracker.update_state(di_loss)
        # self.gen_insulation_tracker.update_state(ins_loss)
        self.gen_grad_tracker.update_state(g_gradient_loss)
        self.gen_symm_tracker.update_state(symm_loss) # For Tpu Training

        return {
            # "gen_ins_loss": self.gen_insulation_tracker.result(),
            "psnr_loss": self.psnr_loss_tracker.result(),
            "mse_loss": self.mse_loss_tracker.result(),
            # "gen_symm_loss": self.gen_symm_tracker.result(),
            # "gen_gradient_loss": self.gen_grad_tracker.result(),
            }

    def predict_step(self, data):
        fake_imgs = self(data)[0] #Fake images
        return fake_imgs

custom_objects = {"ModelWrapper": ModelWrapper,
                    "create_gen_lite": create_gen_lite,
                    "create_gen": create_gen,
                    "u2net": u2net,
                    "RSU_4F": RSU_4F,
                    "RSU_L": RSU_L,
                    "conv_block": conv_block,
                    "TransformerBlock2D":TransformerBlock2D,
                    "MultiHeadAttnBlock2D": MultiHeadAttnBlock2D,
                  }


## Model creation
def create_model(args):
    model = ModelWrapper(input_shape=args["input_shape"],
                            global_batch_size=args["batch_size"],
                            n_heads=args["num_heads"],
                            symmetry_loss_weight=0,
                            insulation_loss_weight=0,
                            gen_gradient_loss_weight=0,
                            mse_loss_weight=1, psnr_loss_weight=1)
    
    # optimizer = tf.keras.optimizers.Adam(args["lr"])
    optimizer = tfa.optimizers.LAMB(learning_rate=args["lr"])
    model.compile(optimizer)
    return model


def create_callbacks(metric="mse_loss", save_path="."):
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=metric,
        mode='auto',
        factor=0.5,
        patience=7,
        min_delta=1e-7,
        verbose=0
    )

    earlystop = tf.keras.callbacks.EarlyStopping(
        monitor=f"val_{metric}",
        mode='auto',
        patience=50,
        verbose=1,
        restore_best_weights=True
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path,
        monitor=metric,
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch',
    )

    callbacks = [
        reducelr,
        earlystop,
        # checkpoint
    ]

    return callbacks


def get_dataset(lr_set, batch_size, strategy, hr_set=None, training=False):
    AUTO = tf.data.AUTOTUNE
    if hr_set is not None:
        dataset = tf.data.Dataset.from_tensor_slices((lr_set, hr_set))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((lr_set))

    if training and hr_set is not None:
        dataset = dataset.shuffle(lr_set.shape[0], reshuffle_each_iteration=True)
        dataset = dataset.repeat()
    dataset = dataset.prefetch(AUTO)\
                    .batch(batch_size, drop_remainder=training, num_parallel_calls=AUTO)
    # This part is for multi-gpu servers
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.FILE
    dataset = dataset.with_options(options)
    dataset = strategy.experimental_distribute_dataset(dataset)
    return dataset



def create_directories(save_dir,
                       models_dir="models",
                       outputs="out") -> None:
    for dd in [save_dir,
               f"{save_dir}/{models_dir}",
               f"{save_dir}/{outputs}"]:
        if not os.path.exists(dd):
            os.makedirs(dd)
    pass


def clear_dir(path) -> None:
    # credit: https://stackoverflow.com/a/72982576/4260559
    if os.path.exists(path):
        for entry in os.scandir(path):
            if entry.is_dir():
                clear_dir(entry)
            else:
                os.remove(entry)
        os.rmdir(path)  # if you just want to delete the dir content but not the dir itself, remove this line

def map_pos_matrics_to_values(val_tuple):
    return [int(val_tuple[1][3:])]

def load_training_data(input_dir, ratio):
    # hr_valid = np.load(f"{input_dir}/{ratio}_ratio/hr_valid.npy").astype("float32")
    # lr_valid = np.load(f"{input_dir}/{ratio}_ratio/lr_valid_ratio{ratio}.npy").astype("float32")

    hr_train = np.load(f"{input_dir}/{ratio}_ratio/hr_train.npy").astype("float32")
    lr_train = np.load(f"{input_dir}/{ratio}_ratio/lr_train_ratio{ratio}.npy").astype("float32")

    chr_numbers = None
    if os.path.exists(f"{input_dir}/{ratio}_ratio/distance_train.npy"):
        chr_numbers = np.load(f"{input_dir}/{ratio}_ratio/distance_train.npy")
        chr_numbers = np.apply_along_axis(map_pos_matrics_to_values, -1, chr_numbers).astype("int32")

    train_indices, valid_indices = train_test_split(np.arange(len(hr_train)),
                                    test_size=.1,
                                    random_state=2025,
                                    shuffle=True,
                                    stratify=chr_numbers
                                    )
    
    hr_valid, lr_valid = hr_train[valid_indices], lr_train[valid_indices]
    hr_train, lr_train = hr_train[train_indices], lr_train[train_indices]
    # Normalize outliers
    # hr_train_cutoff = np.percentile(hr_train, 99.9)
    # # hr_valid_cutoff = np.percentile(hr_valid, 99.9)
    # hr_train = np.minimum(hr_train, hr_train_cutoff)
    # hr_valid = np.minimum(hr_valid, hr_train_cutoff)
    # lr_train = np.minimum(lr_train, hr_train_cutoff)
    # lr_valid = np.minimum(lr_valid, hr_train_cutoff)

    # #Scale values
    # max_hr_train = np.max(hr_train)
    # # max_hr_valid = np.max(hr_valid)
    # hr_train = hr_train/max_hr_train
    # lr_train = lr_train/max_hr_train
    # hr_valid = hr_valid/max_hr_train
    # lr_valid = lr_valid/max_hr_train

    return lr_train, hr_train, lr_valid, hr_valid, 1.0

def load_test_data(input_dir, ratio, hr_train_cutoff):
    if "to_predict" not in input_dir:
        lr_test = np.load(f"{input_dir}/{ratio}_ratio/lr_test_ratio{ratio}.npy").astype("float32")
        # lr_test = np.moveaxis(lr_test, 1, -1)
        # Normalize outliers
        # lr_test = np.minimum(lr_test, hr_train_cutoff)

        #Scale values
        # max_hr_test = np.max(lr_test)
        lr_test = lr_test/hr_train_cutoff
        return {f"lr_test_ratio{ratio}":lr_test}
    else:
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and "hr_" not in f and "pos" not in f]
        outputs = {}
        for file_name in files:
            lr_test = np.load(os.path.join(input_dir, file_name)).astype("float32")
            # lr_test = np.minimum(lr_test, hr_train_cutoff)
            #Scale values
            # max_hr_test = np.max(lr_test)
            lr_test = lr_test/hr_train_cutoff
            outputs[file_name.split(".")[0]] = lr_test
        return outputs
    
    

def train_the_model(args) -> None:
    if args.restart_training:
        clear_dir(args.save_dir)

    NUM_EPOCHS = args.epochs
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    N_REPLICAS = strategy.num_replicas_in_sync
    pprint(f"Num gpus to be used: {N_REPLICAS}")
    BATCH_SIZE = args.batch_size_per_gpu * N_REPLICAS

    create_directories(args.save_dir)
    
    lr_train, hr_train, lr_valid, hr_valid, hr_train_cutoff = load_training_data(args.input, args.ratio)
    args.train_cutoff = hr_train_cutoff

    with open(f"{args.save_dir}/commandline_args.json", 'w') as f:
        json.dump(args.__dict__, f, indent=4)

    train_data_shape = lr_train.shape
    steps_per_epoch = len(lr_train) // BATCH_SIZE
    validation_steps = len(lr_valid) // BATCH_SIZE

    
    train_dataset = get_dataset(lr_train, BATCH_SIZE,
                                strategy=strategy,
                                hr_set=hr_train,
                                training=True)
    valid_dataset = get_dataset(lr_valid, BATCH_SIZE,
                                strategy=strategy,
                                hr_set=hr_valid,
                                training=False)
    del lr_valid, hr_valid, lr_train, hr_train
    K.clear_session()
    callbacks = create_callbacks(save_path=f"{args.save_dir}/models/checkpoints/cp.ckpt")
    model_args = {
        "input_shape": train_data_shape[1:],
        "batch_size": BATCH_SIZE,
        "num_heads": args.na_heads,
        "lr": args.lr
    }
    with strategy.scope():
        model = create_model(model_args)
        if args.model_summary:
            model.generator.summary()
        history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch,
                            epochs=NUM_EPOCHS,
                            validation_data=valid_dataset,
                            validation_steps=validation_steps,
                            callbacks=callbacks, verbose=2)
        model.save(f"{args.save_dir}/models/final_model.ckpt")
    pass


def impute_the_target(args):
    if os.path.exists(f"{args.save_dir}/commandline_args.json"):
        with open(f"{args.save_dir}/commandline_args.json", 'r') as f:
            training_args = json.load(f)
        # Ensure that the same cut-off of the training set is used for dealing with outlier reads
        args.train_cutoff = training_args["train_cutoff"]

    BATCH_SIZE = args.batch_size_per_gpu  # * N_REPLICAS
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    N_REPLICAS = strategy.num_replicas_in_sync
    pprint(f"Num gpus to be used: {N_REPLICAS}")
    BATCH_SIZE = args.batch_size_per_gpu * N_REPLICAS

    K.clear_session()
    model = tf.keras.models.load_model(
        f"{args.save_dir}/models/final_model.ckpt",
        custom_objects=custom_objects,
        compile=False
    )

    lr_test_dict = load_test_data(args.input, args.ratio, args.train_cutoff)
    for keey, vall in lr_test_dict.items():
        print(f"Predicting {keey} with shape of {vall.shape}")

        AUTO = tf.data.AUTOTUNE
        test_dataset = tf.data.Dataset.from_tensor_slices((vall))\
                    .prefetch(AUTO)\
                    .batch(BATCH_SIZE, drop_remainder=False, num_parallel_calls=AUTO)
        steps = int(np.ceil(vall.shape[0]/BATCH_SIZE))
        test_preds = np.round(model.generator.predict(test_dataset, steps=steps)[0]*args.train_cutoff).astype(np.float32)
        # for ii in range(len(test_preds)):
        #     print(f"Output[{ii}] shape: {test_preds[ii].shape}")
        # with gzip.GzipFile(f"{args.save_dir}/out/test_preds.npy.gz", "w") as f:
        np.save(f"{args.save_dir}/out/preds_{keey}.npy", arr=test_preds)
        # np.save(f"{args.save_dir}/out/test_preds", test_preds)
        filesize = os.path.getsize(f"{args.save_dir}/out/preds_{keey}.npy") >> 20
        print(f"The file size is {filesize}MBs")

def str_to_bool(s):
    # Define accepted string values for True and False
    true_values = ['true', '1']
    false_values = ['false', '0']

    # Convert the input string to lowercase for case-insensitive comparison
    lower_s = s.lower()

    # Check if the input string is in the list of true or false values
    if lower_s in true_values:
        return True
    elif lower_s in false_values:
        return False
    else:
        raise ValueError(f"Invalid boolean value: {s}. Accepted values are 'true', 'false', '0', '1'.")


def main():
    deciding_args_parser = argparse.ArgumentParser(description='ShiLab\'s TruHiC.', add_help=False)

    ## Function mode
    deciding_args_parser.add_argument('--mode', type=str, help='Operation mode: denoise | train (default=train)',
                                      choices=['enhance', 'train'], default='train')
    deciding_args_parser.add_argument('--restart-training', type=str, required=False,
                                      help='Whether to clean previously saved models in target directory and restart the training',
                                      choices=['false', 'true', '0', '1'], default='0')
    deciding_args, _ = deciding_args_parser.parse_known_args()
    parser = argparse.ArgumentParser(
        description="", parents=[deciding_args_parser])
    ## Input args
    parser.add_argument('--input', type=str, required=True, help='Input directory path.')
    parser.add_argument('--ratio', type=str, required=True, help='Downsampling ratio.')
    # parser.add_argument('--target', type=str, required=(deciding_args.mode != 'train'),
    #                     help='Target file path. Must be provided in "enhance" mode.')

    ## save args
    parser.add_argument('--save-dir', type=str, required=True, help='the path to save the results and the model.\n'
                                                                    'This path is also used to load a trained model for enhancement.')
    ## Model (hyper-)params
    parser.add_argument('--model-summary', type=str, required=False,
                                      help='Prints the model summary.',
                                      choices=['false', 'true', '0', '1'], default='0')
    parser.add_argument('--val-frac', type=float, required=False,
                        help='Fraction of reference samples to be used for validation (default=0.1).', default=0.1)
    parser.add_argument('--epochs', type=int, required=False, help='Maximum number of epochs (default 1000)',
                        default=1000)
    parser.add_argument('--na-heads', type=int, required=False, help='Number of attention heads (default 16)',
                        default=16)
    parser.add_argument('--lr', type=float, required=False, help='Learning Rate (default 0.001)', default=0.001)
    parser.add_argument('--batch-size-per-gpu', type=int, required=False, help='Batch size per gpu(default 16)',
                        default=16)

    # misc
    parser.add_argument('--verbose', type=int, required=False,
                        help='Training verbosity', default=2)
    
    args = parser.parse_args()
    args.restart_training = str_to_bool(args.restart_training)
    args.model_summary = str_to_bool(args.model_summary)

    if not (args.save_dir.startswith("./") or args.save_dir.startswith("/")):
        args.save_dir = f"./{args.save_dir}"
    pprint(f"Save directory will be:\t{args.save_dir}")

    if args.mode == 'train':
        train_the_model(args)
    else:
        impute_the_target(args)


if __name__ == '__main__':
    main()
