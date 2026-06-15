import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K

keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable(package="MyLayers", name="main_module")
def dfhic(input_shape):
    """ Input Layer """
    inputs = layers.Input(input_shape)
    num_channels = input_shape[-1]
    w_init = tf.keras.initializers.RandomNormal(stddev=0.02)
    n_0 = layers.Conv2D(32, (3, 3), dilation_rate=1, padding='SAME', activation='relu', kernel_initializer=w_init)(inputs)
    n_1 = layers.Conv2D(32, (3, 3), dilation_rate=2, padding='SAME', activation='relu', kernel_initializer=w_init)(n_0)
    n_2 = layers.Conv2D(32, (3, 3), dilation_rate=2, padding='SAME', activation='relu', kernel_initializer=w_init)(n_1)
    n_3 = n_0 + n_2  # Element-wise addition (skip connection)
    n_4 = layers.Conv2D(64, (3, 3), dilation_rate=2, padding='SAME', activation='relu', kernel_initializer=w_init)(n_3)
    n_5 = layers.Conv2D(64, (3, 3), dilation_rate=2, padding='SAME', activation='relu', kernel_initializer=w_init)(n_4)
    n_6 = n_4 + n_5  # Element-wise addition (skip connection)
    n_7 = layers.Conv2D(128, (3, 3), dilation_rate=2, padding='SAME', activation='relu', kernel_initializer=w_init)(n_6)
    n_8 = layers.Conv2D(128, (3, 3), dilation_rate=2, padding='SAME', activation='relu', kernel_initializer=w_init)(n_7)
    n_9 = n_7 + n_8  # Element-wise addition (skip connection)
    n_10 = layers.Conv2D(256, (3, 3), dilation_rate=2, padding='SAME', activation='relu', kernel_initializer=w_init)(n_9)
    n_11 = layers.Conv2D(256, (3, 3), dilation_rate=2, padding='SAME', activation='relu', kernel_initializer=w_init)(n_10)
    n = layers.Conv2D(1, (1, 1), dilation_rate=1, padding='SAME', activation='relu', kernel_initializer=w_init)(n_11)
    print(f"n.shape:", n.shape)
    n += inputs[..., 0:1]  # Final skip connection

    model = tf.keras.models.Model(inputs, outputs=n)
    return model


@keras.saving.register_keras_serializable(package="MyModels")
class ModelWrapper(tf.keras.Model):
    def __init__(self):
        super(ModelWrapper, self).__init__()

    def build(self, input_shape):
        self.m_module = dfhic(input_shape[1:])

    def call(self, inputs):
        return self.m_module(inputs)
    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "m_module": self.m_module,
            }
        )
        return config

    
    