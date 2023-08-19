"""
Pipeline
--------


A Set of algorithms for cleaning the textual data into
something that is easier for the computer to read and
analyze.
"""
import re
import numpy as np
from keras.utils import np_utils

def _main():
    pass


def clean_text(text:str, pad:str = '<pad>') -> str:
    """
    Clean Text to ASCII standard.
    -----------------------------

    Remove any special characters that do not add or
    provide context within the text. Also organizes
    sentences such that they do not possess any extra
    spaces and allows for proper context in terms of
    punctuation.

    Parameters
    ----------
    text : String
        Text in its rawest form, created by simply loading
        the text from the .txt file. This text should be
        loaded in using UTF-8 encoding.

    pad : String
        Padding to separate sample stories within the text.
    """
    text = text.lower()
    text = pad + text
    text = re.sub(r'\n{4,}', pad, text)
    text = text.replace('\n\n', '. ')
    text = text.replace('\n', ' ')
    text = text.replace(".'.", ".'")
    text = text.replace('!".', '"')
    text = text.strip()
    text = text.replace("..", ".")
    text = re.sub(r'([!"#$%&()*+,-./:;<=>?@[\]^_`{|}~])', r' \1 ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.replace(' . ', '. ')
    return text

def create_vocabulary(text:str, delimiter:str=" ", pad:str="<pad>") -> list, dict:
    """
    Create a dictionary containing all of the words within
    the raw raw text indexed by numbers.
    ...

    Separates the words found within the clean body of text
    based on a predesignated delimiter (such as the space
    between the words) and assigns a number to the word.

    Parameters:
    -----------
    text : String
        Text after light processing. This is a singular
        string which has been modified to exclude special
        characters or anything that might not be consistent
        with english writing convention.

    delimiter : String
        Key used to separate the text into its separate
        entities or tokens.
    """
    tokens = text.split(delimiter)
    tokens.append(' ')
    unique_tokens = sorted(set(tokens))
    vocabulary = {i:unique_tokens[i] for i in range(len(unique_tokens))}
    return tokens, vocabulary

def generate_sequences(tokens:list, sequence_length:int, step:int):
    """
    Generate the dataset based on sequences of words.
    ...

    Uses a list of tokens to create the dataset by indexing
    the tokens based on sequence length.

    Parameters
    ----------
    tokens : List
        List containing words in the order of the story.

    sequence_length: Integer
        The distance into the future the model will predict.

    step : Integer
        The number of steps taken.
    """
    X = []
    y = []

    for i in range(0, len(tokens) - sequence_length, step):
        X.append(tokens[i:i + sequence_length])
        y.append(tokens[i + sequence_length])

    y = np_utils.to_categorical(y, num_classes = set(tokens))
    num_seq = len(X)
    return X, y, num_seq


if __name__ == "__main__":
    _main()
