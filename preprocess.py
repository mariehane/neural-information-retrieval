import tqdm
import nltk
import spacy
import lemminflect
import pandas as pd

def _lemmatize_apply(texts, function, show_progressbar):
    if show_progressbar:
        tqdm.tqdm.pandas()
        return texts.progress_apply(function)
    return texts.apply(function)


def lemmatize_wordnet(texts, show_progressbar=False):
    wordnet = nltk.WordNetLemmatizer()
    return _lemmatize_apply(texts, wordnet.lemmatize, show_progressbar)

def stem_porter(texts, show_progressbar=False):
    porter = nltk.PorterStemmer()
    return _lemmatize_apply(texts, porter.stem, show_progressbar)

def stem_snowball(texts, show_progressbar=False):
    snowball = nltk.SnowballStemmer('english')
    return _lemmatize_apply(texts, snowball.stem, show_progressbar)
    

def lemmatize_lemminflect(texts, show_progressbar=False):
    nlp = spacy.load("en_core_web_sm")
    lemm_inflect = lambda doc: " ".join(token._.lemma() for token in doc)
    
    iterator = nlp.pipe(texts)

    if show_progressbar:
        iterator = tqdm.tqdm(iterator, total=len(texts), smoothing=0.01)
    
    return pd.Series(lemm_inflect(doc) for doc in iterator)
