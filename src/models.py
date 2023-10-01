"""

Creative Models.

----------------

Set of classes or functions that are used to develop
generative models.
"""
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adam, Adafactor
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.saved_model import load
import numpy as np
import random
from pipeline import clean_text, generate_sequences, version, HangmanEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments import BatchedPyEnvironment

EPOCHS = 1_000
BATCH= 32

def _main():
    policy = load("models/policy_{}".format(version))
    env = HangmanEnvironment(["hello"])
    tenv = BatchedPyEnvironment([env])
    env.add_guessed_letters("_ _ _ _ _ _ _ _ _") # goodnight
    #print(env._guessed_letters)
    #print(env._state)
    #exit()
    test_env = TFPyEnvironment(env)
    time_step = test_env.reset()
    for _ in range(20):
        tstep = policy.action(time_step)
        time_step = test_env.step(tstep.action[0])
        print("episode: {}, result: {}".format(_,tstep.action[0]))
        print(time_step)


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


if __name__ == "__main__":
    _main()
