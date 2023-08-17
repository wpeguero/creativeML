"""
Pipeline
--------


A Set of algorithms for cleaning the textual data into
something that is easier for the computer to read and
analyze.
"""
import re

def _main():
    pass


def _clean_text(text:str, pad:str = '<pad>') -> str:
    """
    Clean Text to ASCII standard.
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


if __name__ == "__main__":
    _main()
