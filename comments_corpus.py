import pandas as pd
from pymongo import MongoClient
import pickle
import csv,re,sys,spacy
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from string import punctuation,printable
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

def data_to_pd():
    client = MongoClient()
    db = client.bgg
    df = pd.DataFrame(list(db.game_comments.find()))
    return df

def comments_to_corpus(df):
    game_ids = list(df.game_id.unique())
    corpus = []
    for id in game_ids:
        corpus.append(' '.join(df[(df['game_id'] == id)]['comment'].tolist()))
        print(len(corpus))
    return corpus

def corpus_to_pkl(corpus):
    with open('data/comments_corpus.pkl', 'wb') as fp:
        pickle.dump(corpus, fp)

def make_corpus():
    df = data_to_pd()
    print("df_created")
    corpus = comments_to_corpus(df)
    corpus_to_pkl(corpus)
    print("corpus is pickled")

def load_comments_corpus():
    with open('data/comments_corpus.pkl', 'rb') as fp:
        comments_corpus = pickle.load(fp)
    return comments_corpus

def tokenize_string(doc, nlp):
    clean_doc = ""
    for char in doc:
        if char in printable:
            clean_doc += char
    # Run the doc through spaCy
    tokenized_doc = nlp(clean_doc)
    return tokenized_doc

def lemmatize_string(doc):
    # Lemmatize and lower text
    lem_doc = " ".join([re.sub("\W+","",token.lemma_.lower()) if token.lemma_ != "-PRON-" else token.lower_ for token in doc])
    return lem_doc

def remove_stopwords(doc, stop_words):
    no_stop = " ".join([token for token in doc.split() if token not in stop_words])
    return no_stop

def to_pickle(doc_lst):
    with open('data/comments_doc_lst.pkl', 'wb') as fp:
        pickle.dump(doc_lst,fp)

def un_pickle():
    with open('data/comments_doc_lst', 'rb') as fp:
        processed = pickle.load(fp)
    return processed

def count_vec(processed):
    countvect = CountVectorizer()
    count_vectorized = countvect.fit_transform(processed)
    return count_vectorized
    #vocab = countvect.get_feature_names()

def tfideffed(processed):
    tfidfvect = TfidfVectorizer()
    tfidf_vectorized = tfidfvect.fit_transform(processed)
    return tfidf_vectorized

def tfidf_to_pickle(doc_lst):
    with open('data/desc_tfidf.pkl', 'wb') as fp:
        pickle.dump(tfidf_vectorized, fp)

def cos_sim(tfidf_vectorized):
    cosine_similarities = linear_kernel(tfidf_vectorized, tfidf_vectorized)
    return cosine_similarities

def process():
    #load all descriptions from mongo
    comments_corpus = load_comments_corpus()
    #initiate stopwords
    nlp = spacy.load('en')
    STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m"] + list(ENGLISH_STOP_WORDS))
    #tokenize all descriptions
    token_docs = [tokenize_string(doc, nlp) for doc in comments_corpus]
    print("tokenized")
    #lemmatize
    lemmatized = [lemmatize_string(doc) for doc in token_docs]
    print("lemmatized")
    #remove stopwords
    no_stop = [remove_stopwords(doc, STOPLIST) for doc in lemmatized]
    print("stop words removed")
    #pickle so we don't have to wait on tokenizing
    to_pickle(no_stop)

if __name__ == '__main__':
    process()
