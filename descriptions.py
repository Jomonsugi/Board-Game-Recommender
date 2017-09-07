import csv,re,sys,spacy
from pymongo import MongoClient
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from string import punctuation,printable
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np
from sklearn.metrics.pairwise import linear_kernel


def load_documents():
    client = MongoClient()
    db = client.bgg
    coll = db.game_stats
    corpus = [game_stats['description'] for game_stats in coll.find()]
    return corpus

def tokenize_string(doc):
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
    with open('data/doc_lst.pkl', 'wb') as fp:
        pickle.dump(doc_lst,fp)

def un_pickle():
    with open('data/doc_lst.pkl', 'rb') as fp:
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
    vocabulary = np.array(tfidfvect.get_feature_names())
    return tfidf_vectorized, vocabulary

def tfidf_to_pickle(doc_lst):
    with open('data/desc_tfidf.pkl', 'wb') as fp:
        pickle.dump(tfidf_vectorized, fp)

def cos_sim(tfidf_vectorized):
    cosine_similarities = linear_kernel(tfidf_vectorized, tfidf_vectorized)
    return cosine_similarities

def process():
    #load all descriptions from mongo
    corpus = load_documents()
    #initiate stopwords
    nlp = spacy.load('en')
    STOPLIST = set(stopwords.words('english') + ["n't", "'s", "'m"] + list(ENGLISH_STOP_WORDS))
    #tokenize all descriptions
    token_docs = [tokenize_string(doc) for doc in corpus]
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
    processed = un_pickle()
    #note that count_vec is done in the tfidf step below (combining a count and tranform step, but this way I have the vocab list if needed and the actual counts)
    # count_mat = count_vec(processed)
    #vocab = countvect.get_feature_names()
    #to produce a term frequency index document frequncy matrx
    tfidf_vectorized, vocabulary = tfideffed(processed)
    #find cosine similarities
    # cosine_similarities = cos_sim(tfidf_vectorized)
    #pkl tfidf
    # tfidf_to_pickle(tfidf_vectorized)
