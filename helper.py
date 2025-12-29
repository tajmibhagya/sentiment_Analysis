import numpy as np
import pandas as pd
import re
import string
import pickle
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

ps = PorterStemmer()
stopwords = set(ENGLISH_STOP_WORDS)

# Load vocabulary
vocabulary = pd.read_csv('static/model/vocabulary.txt', header=None)[0].tolist()

def remove_stopwords(text):
    return " ".join(word for word in text.split() if word.lower() not in stopwords)

def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(re.sub(r'(https?://\S+|www\.\S+)', ' ', x, flags=re.MULTILINE) for x in x.split()))
    data['tweet'] = data['tweet'].apply(remove_punctuations)
    data['tweet'] = data['tweet'].str.replace(r'\d+', '', regex=True)
    data['tweet'] = data['tweet'].apply(remove_stopwords)
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(ps.stem(word) for word in x.split()))
    return data['tweet']

def vectorizer(ds):
    vectorized_lst = []
    for sentence in ds:
        sentence_lst = np.zeros(len(vocabulary))
        for i in range(len(vocabulary)):
            if vocabulary[i] in sentence.split():
                sentence_lst[i] = 1
        vectorized_lst.append(sentence_lst)
    return np.asarray(vectorized_lst, dtype=np.float32)

def get_prediction(vectorized_txt):
    prediction = model.predict(vectorized_txt)
    
    if prediction[0] == 1:
        return 'Negative'
    else:
        return 'Positive'
