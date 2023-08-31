"""

Creative Models.

----------------

Set of classes or functions that are used to develop
generative models.
"""
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import RMSprop, Nadam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
import numpy as np
import random
from pipeline import clean_text, generate_sequences, version

EPOCHS = 130
BATCH= 32

def _main():
    filename = "./data/aesop_tales.txt"
    with open(filename, encoding='utf-8') as f:
        text = f.read()
    seq_length = 20
    start_story = "| " * seq_length
    start = text.find("THE FOX AND THE GRAPES\n\n\n")
    end = text.find("ILLUSTRATIONS\n\n\n[")
    text = text[start:end]
    text = clean_text(text)
    # Tokenization
    tokenizer = Tokenizer(char_level=False, filters='')
    tokenizer.fit_on_texts([text])
    total_words = len(tokenizer.word_index) + 1
    token_list = tokenizer.texts_to_sequences([text])[0]
    print(total_words)
    # Parameters
    n_units = 512
    embedding_size = 220
    # build Dataset
    X, y, num_seq = generate_sequences(token_list,sequence_length=seq_length, step=1)
    # Build Model
    model = TextGenerator(total_words, embedding_size, n_units)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    plot_model(model, to_file='models/diagrams/model_v{}.png'.format(version), show_shapes=True, show_dtype=True)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH, shuffle=True)
    model.save('models/model_v{}'.format(version))

def TextGenerator(total_words:int, embedding_size:int, n_units:int):
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
    x = LSTM(n_units, return_sequences=True)(x)
    x = LSTM(int(n_units * 4))(x)
    x = Dropout(0.3)(x)
    text_out = Dense(total_words, activation='softmax')(x)
    model = Model(text_in, text_out)
    return model


if __name__ == "__main__":
    _main()
