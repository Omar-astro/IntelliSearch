import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


stemmer = PorterStemmer()

stop_words = set(stopwords.words('english'))


def preprocess(text):

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    tokens = word_tokenize(text)

    processed_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words
    ]

    return ' '.join(processed_tokens)


def preprocess_elmo(text):

    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text