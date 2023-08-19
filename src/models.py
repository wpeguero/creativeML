"""

Creative Models.

----------------

Set of classes or functions that are used to develop
generative models.
"""
from keras.layers import Dense, LSTM, Input, Embedding, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
import numpy as np
import random

def _main():
    pass


class TextGenerator(Model):
    """

    Model For Designing a Text Generator.

    ...

    Model designed to develop a text generator.
    This class will allow one to develop models for
    developing interesting story-tellers based on books
    written by famous authors.
    """

    def __init__(self, vocab_size, embedding_dim, rnn_units):
        """

        Initialize the class.

        ...

        Parameters
        ----------
        vocab_size : Integer
            The number of unique tokens found within the
            corpus.

        embedding_dim : Integer
            The number of dimensions within the embedding layer.

        rnn_units : Integer
            The number of separate LSTM units within the LSTM layer.
        """
        super().__init__(self)
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm1 = LSTM(rnn_units, return_sequences=True)
        self.lstm2 = LSTM(int(rnn_units/2), return_sequences=True)
        self.dropout = Dropout(0.2)
        self.dense = Dense(vocab_size, activation="softmax")

    def call(self, inputs, training=False):
        """Call upon the layers of the custom model for training."""
        x = inputs
        x = self.embedding(x, training=training)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dropout(x)
        outputs = self.dense(x)
        return outputs


if __name__ == "__main__":
    _main()
