"""
Pipeline.

--------


A Set of algorithms for cleaning the textual data into
something that is easier for the computer to read and
analyze.
"""
import re
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.models import load_model
import json

PAD = '| '

def _main():
    model = load_model("models/model_v18")
    word = "The frog and the fox."
    next_words = 500
    filename = "./data/aesop_tokenizer.json"
    with open(filename, encoding='utf-8') as f:
        json_tokenizer = json.load(f)
        f.close()
    tokenizer = tokenizer_from_json(json_tokenizer)
    text = generate_text(word, next_words, model, tokenizer, temp=0.2)
    print(text)


def clean_text(text:str, pad:str = '|') -> str:
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
    text = re.sub(r'\n{4,}', pad, text) # Need to change this
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

def create_vocabulary(text:str, delimiter:str=" ", pad:str="<pad>"):
    """
    Create a dictionary containing all of the words within the raw raw text indexed by numbers.

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

    for i in range(0, len(set(tokens)) + 1 - sequence_length, step):
        X.append(tokens[i:i + sequence_length])
        y.append(tokens[i + sequence_length])

    y = to_categorical(y, num_classes = len(set(tokens)) + 1)
    num_seq = len(X)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y, num_seq

def sample_with_temp(preds, temperature:float=1.0):
    """Sample an index from a probability array."""
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)

def generate_text(seed_text:str, next_words:int, model, tokenizer:Tokenizer, msl:int=20, temp:float=1.0) -> str:
    """
    Generate a body of text.

    ...

    Uses a generative machine learning model to create a
    story based on the a predetermined set of initial words
    and the total number of words desired. The determined
    number of words does not mean that the story will be as
    long, but rather that the max possible number of words
    will be equivalent to the variable `next words`. The
    msl allows the model to read the data in steps of n.

    Parameters
    ----------
    seed_text : String
        The beginning of the story. Predetermined by the
        user of the machine learning model.

    next_words : Integer
        The maximum possible number of words that the
        machine learning model will generate.

    model : TensorFlow Model
        Machine learning model used to generate text. Input
        text and the model will predict the next possible
        words.

    tokenizer : TensorFlow Tokenizer
        Model which converts tokens (words) to sequences.
        Part of TensorFlow's preprocessing algorithms.

    msl : Integer
        msl (Max Sequence Length) limits the length of the
        input to the machine learning model.

    temp : Floating Point Number
        Measure for the helper function to sample an index
        from a probability array.

    Returns
    -------
    output_text : String
        The complete story generated by the model.
    """
    output_text = seed_text
    start_story = PAD * msl
    seed_text = start_story + seed_text

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-msl:]
        token_list = np.reshape(token_list, (1, msl))

        probs = model.predict(token_list, verbose=0)[0]
        y_class = sample_with_temp(probs, temp)

        output_word = tokenizer.index_word[y_class] if y_class > 0 else ''

        if output_word == "|":
            break

        seed_text += output_word + ' '
        output_text += output_word + ' '

    return output_text


if __name__ == "__main__":
    _main()
