"""

Creative Models.

----------------

Set of classes or functions that are used to develop
generative models.
"""
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
import numpy as np
import random
from pipeline import clean_text, generate_sequences

EPOCHS = 1000
BATCH= 32
version=2

def _main():
    filename = "./data/aesop_tales.txt"
    with open(filename, encoding='utf-8') as f:
        text = f.read()
    seq_length = 20
    start_story = "|" * seq_length
    start = text.find("THE FOX AND THE GRAPES\n\n\n")
    end = text.find("ILLUSTRATIONS\n\n\n[")
    text = text[start:end]
    text = clean_text(text)
    # Tokenization
    tokenizer = Tokenizer(char_level=False, filters='')
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    token_list = tokenizer.texts_to_sequences([text])[0]
    # Parameters
    n_units = 256
    embedding_size = 100
    # build Dataset
    X, y, num_seq = generate_sequences(token_list,sequence_length=seq_length, step=1)
    print(X.shape)
    print(y.shape)
    # Build Model
    text_in = Input(shape=(None,))
    x = Embedding(total_words, embedding_size)(text_in)
    x = LSTM(n_units)(x)
    x = Dropout(0.2)(x)
    text_out = Dense(total_words, activation='softmax')(x)
    model = Model(text_in, text_out)
    #model = TextGenerator(total_words, embedding_size, n_units)
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss='categorical_crossentropy')
    #model.build(input_shape=(None,))
    plot_model(model, to_file='models/diagrams/model_v{}.png'.format(version), show_shapes=True, show_dtype=True)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, shuffle=True)
    model.save('models/model_v{}'.format(version))


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
        super(TextGenerator, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.lstm1 = LSTM(rnn_units)
        self.lstm2 = LSTM(int(rnn_units/2))
        self.dropout = Dropout(0.2)
        self.dense = Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        """Call upon the layers of the custom model for training."""
        x = inputs
        x = self.embedding(x)
        x = self.lstm1(x)
        x = self.lstm2(x)
        x = self.dropout(x)
        outputs = self.dense(x)
        return outputs


if __name__ == "__main__":
    _main()
