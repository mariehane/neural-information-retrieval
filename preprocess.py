import tqdm
import nltk
import spacy
import lemminflect
import numpy as np
import pandas as pd
from datetime import datetime

def _apply(texts, function, show_progressbar):
    if show_progressbar:
        tqdm.tqdm.pandas()
        return texts.progress_apply(function)
    return texts.apply(function)

def remove_stopwords(texts, show_progressbar=False):
    stopwords = nltk.corpus.stopwords.words('english')
    f = lambda text: " ".join(w for w in text.split(" ") if w.lower() not in stopwords)
    return _apply(texts, f, show_progressbar)

def lemmatize_wordnet(texts, show_progressbar=False):
    wordnet = nltk.WordNetLemmatizer()
    return _apply(texts, wordnet.lemmatize, show_progressbar)

def stem_porter(texts, show_progressbar=False):
    porter = nltk.PorterStemmer()
    return _apply(texts, porter.stem, show_progressbar)

def stem_snowball(texts, show_progressbar=False):
    snowball = nltk.SnowballStemmer('english')
    return _apply(texts, snowball.stem, show_progressbar)
    

def lemmatize_lemminflect(texts, show_progressbar=False):
    #nlp = spacy.load("en_core_web_sm")
    def lemmatize_word(word):
        lemmas = lemminflect.getLemma(word, 'NOUN', lemmatize_oov=False) 
        if len(lemmas) != 0:
            return lemmas[0]
        lemmas = lemminflect.getLemma(word, 'VERB', lemmatize_oov=False)
        if len(lemmas) != 0:
            return lemmas[0]
        else:
            return lemminflect.getLemma(word, 'NOUN', lemmatize_oov=True)[0]
    lemm_inflect = lambda doc: " ".join(lemmatize_word(word) for word in doc.split(" "))
    
    return _apply(texts, lemm_inflect, show_progressbar)
    #iterator = nlp.pipe(texts)

    #if show_progressbar:
    #    iterator = tqdm.tqdm(iterator, total=len(texts), smoothing=0.01)
    
    #return pd.Series(lemm_inflect(doc) for doc in iterator)

def remove_duplicates(df):
    """Duplicate removal that prefers rows with filled title, abstract and text
    """
    df_sorted = df.sort_values(by="title").sort_values(by="abstract").sort_values(by="text")
    return df_sorted.drop_duplicates(subset="cord_uid", keep="first")

def remove_without_title(df):
    return df.dropna(subset=['title'])

def remove_before(df, date=datetime(2000, 1, 1)):
    def parse_date(datestr):
        if datestr is np.nan:
            return np.nan
        try:
            return datetime.fromisoformat(datestr)
        except ValueError:
            return datetime.fromisoformat(datestr+'-01-01')
    df_times = df.publish_time.apply(parse_date)
    return df[df_times >= date]

