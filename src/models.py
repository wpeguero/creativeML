"""

Creative Models.

----------------

Set of classes or functions that are used to develop
generative models.
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout, GRU, Reshape, Flatten, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam, Adafactor
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.saved_model import load
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments import BatchedPyEnvironment

import numpy as np

import random

from pipeline import clean_text, generate_sequences, version, DCGANTrainer

EPOCHS = 1_000
BATCH= 32

def _main():
    model = SimpleGenerator()
    model.build(input_shape=(None, 100))
    discriminator = SimpleDiscriminator()
    discriminator.build(input_shape=(None, 28, 28, 1))
    plot_model(model, to_file='models/diagrams/generator_v{}.png'.format(version))
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5 #Normalize images to [-1, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60_000).batch(256)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    trainer = DCGANTrainer(model, discriminator, loss, BATCH, 100)
    trainer.train(train_dataset, 50)
    gmodel = trainer.generator
    dmodel = trainer.discriminator
    gmodel.save("models/genmodel_{}".format(version))
    dmodel.save("models/dismodel_{}".format(version))


def text_generator(total_words:int, embedding_size:int, n_units:int):
    """

    Model For Designing a Text Generator.

    ...

    Model designed to develop a text generator.
    This class will allow one to develop models for
    developing interesting story-tellers based on books
    written by famous authors.
    """
    text_in = Input(shape=(None,))
    x = Embedding(total_words, embedding_size)(text_in)
    x, states = LSTM(n_units, return_sequences=True)(x)
    x = LSTM(int(n_units * 4), initial_state=states)(x)
    x = Dropout(0.3)(x)
    text_out = Dense(total_words, activation='softmax')(x)
    model = Model(text_in, text_out)
    return model


class TextGenerator(Model):
    """Simple RNN using LSTM layers."""

    def __init__(self, total_words:int, embedding_dim:int, n_units:int):
        """Iniitialize the class."""
        super().__init__(self)
        self.embedding = Embedding(total_words, embedding_dim)
        self.lstm1 = LSTM(n_units, return_sequences=True)
        self.lstm2 = LSTM(int(n_units * 4))
        self.dropout = Dropout(0.3)
        self.dense = Dense(total_words, activation='softmax')

    def call(self, inputs):
        """Call the model for training."""
        x = inputs
        x = self.embedding(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dense(x)
        return x


class SimpleRNN(Model):
    """Simple Recursive Neural Network using Gated Recurrent Units."""

    def __init__(self, total_words:int, embedding_dim:int, n_units:int):
        """Initialize the Class."""
        super().__init__(self)
        self.embedding = Embedding(total_words, embedding_dim)
        self.gru = GRU(n_units, return_sequences=True, return_state=True)
        self.dense = Dense(total_words, activation='softmax')

    def call(self, inputs, states=None, return_state=False, training=False):
        """Call the model for training."""
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dens(x, training=training)
        if return_state:
            return x, states
        else:
            return x


class BasicAutoencoder(Model):
    """
    Simple Autoencoder for converting images.

    -----------------------------------------
    The autoencoder functions by inputting the latent dimensions
    of the image and then proceeding to encode the image. The
    encoded image is then decoded by the autoencoder and the new
    processed image is provided.
    """

    def __init__(self, latent_dim, shape):
        super(BasicAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = Sequential([
            Flatten(),
            Dense(self.latent_dim, activation='relu')
            ])
        self.decoder = Sequential([
            Dense(tf.math.reduce_prod(self.shape), activation='sigmoid'),
            Reshape(self.shape)
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Denoise(Model):
    """Convolutional Autoencoder."""

    def __init__(self, shape:tuple):
        super(Denoise, self).__init__()
        self.shape = shape
        self.encoder = Sequential([
            Input(shape=self.shape),
            Conv2D(16, (3,3), activation='relu', padding='same', strides=2),
            Conv2D(8, (3,3), activation='relu', padding='same', strides=2)
            ])
        self.decoder = Sequential([
            Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SimpleGenerator(Model):
    """Simple Generator."""

    def __init__(self):
        super(SimpleGenerator, self).__init__()
        self.dense1 = Dense(7*7*256, use_bias=False)
        self.batch_norm = BatchNormalization()
        self.leakyrelu = LeakyReLU()
        self.reshape = Reshape((7, 7, 256))
        self.conv2dtranspose1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.batch_norm1 = BatchNormalization()
        self.leakyrelu1 = LeakyReLU()
        self.conv2dtranspose2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batch_norm2 = BatchNormalization()
        self.leakyrelu2 = LeakyReLU()
        self.conv2dtranspose3 = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)

    def call(self, x):
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = self.leakyrelu(x)
        x = self.reshape(x)
        x = self.conv2dtranspose1(x)
        x = self.batch_norm1(x)
        x = self.leakyrelu1(x)
        x = self.conv2dtranspose2(x)
        x = self.batch_norm2(x)
        x = self.leakyrelu2(x)
        x = self.conv2dtranspose3(x)
        return x


class SimpleDiscriminator(Model):
    """CNN-based image classifier."""
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        self.cnn = Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])
        self.leakyrelu = LeakyReLU()
        self.dropout = Dropout(0.3)
        self.cnn1 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.leakyrelu1 = LeakyReLU()
        self.dropout1 = Dropout(0.3)
        self.flatten = Flatten()
        self.dense1 = Dense(1)

    def call(self, x):
        x = self.cnn(x)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.cnn1(x)
        x = self.leakyrelu1(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x


if __name__ == "__main__":
    _main()
