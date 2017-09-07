import pyspark as ps
import pandas as pd
from pymongo import MongoClient
import pickle
import numpy as np
import random
from pyspark.mllib.recommendation import ALS
import math

spark = ps.sql.SparkSession.builder \
            .master("local[2]") \
            .appName("collab_rec") \
            .getOrCreate()

sc = spark.sparkContext

def from_mongo():
    client = MongoClient()
    db = client.bgg
    df = pd.DataFrame(list(db.game_comments.find())).dropna()
    usernames = df['username'].unique()
    usernames_id = pd.Series(np.arange(len(usernames)), usernames)
    df["user_id"] = df['username'].map(usernames_id.get)
    df.to_pickle('data/game_ratings')

def uir_to_list():
    print("uir")
    uir = [(df.iloc[x]["user_id"],df.iloc[x]["game_id"],unicode(df.iloc[x]["rating"])) for x in indices]
    with open('data/uir_500K.pkl', 'wb') as fp:
        pickle.dump(uir, fp)

def ig_to_list():
    print("ig")
    ig = [(df.iloc[x]["game_id"],df.iloc[x]["game"]) for x in indices]
    with open('data/ig_500K.pkl', 'wb') as fp:
        pickle.dump(ig, fp)

def uir_un_pickle():
    with open('data/uir_500K.pkl', 'rb') as fp:
        uir = pickle.load(fp)
    return uir

def ig_un_pickle():
    with open('data/ig_500K.pkl', 'rb') as fp:
        ig = pickle.load(fp)
    return ig

def to_rdd():
    uir_rdd = sc.parallelize(uir_un_pickle(), 8)
    ig_rdd = sc.parallelize(ig_un_pickle(), 8)
    return uir_rdd, ig_rdd

def train_val_test_test():
    training_RDD, validation_RDD, test_RDD = uir_rdd.randomSplit([6, 2, 2], seed=0L)
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
    return training_RDD, validation_RDD, test_RDD, validation_for_predict_RDD, test_for_predict_RDD

def predict_test(training_RDD, validation_RDD, test_RDD, validation_for_predict_RDD, test_for_predict_RDD):
    seed = 5L
    iterations = 10
    regularization_parameter = 0.1
    ranks = [4,8,12]
    errors = [0, 0, 0]
    err = 0
    tolerance = 0.02

    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    for rank in ranks:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                          lambda_=regularization_parameter)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        errors[err] = error
        err += 1
        print 'For rank %s the RMSE is %s' % (rank, error)
        if error < min_error:
            min_error = error
            best_rank = rank

    print 'The best model was trained with rank %s' % best_rank
    return predictions, rates_and_preds

def predict():
    training_RDD, test_RDD = uir_500K_rdd.randomSplit([7, 3], seed=0L)
    complete_model = ALS.train(training_RDD, rank=4, seed=5L,
                           iterations=10, lambda_=0.1)
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

    predictions = complete_model.predictAll(test_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = test_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

    print 'For testing data the RMSE is %s' % (error)

    return predictions, rates_and_preds

def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)

    movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
    movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
    movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

if __name__ == '__main__':
    # from_mongo()
    # df = pd.read_pickle('data/game_ratings')
    # indices = random.sample(range(len(df)), 500000)
    # uir_to_list()
    # ig_to_list()
    uir_500K_rdd, ig_500K_rdd = to_rdd()

    # training_RDD, validation_RDD, test_RDD, validation_for_predict_RDD, test_for_predict_RDD = train_val_test_test()

    # predictions, rates_and_preds = predict_test(training_RDD, validation_RDD, test_RDD, validation_for_predict_RDD, test_for_predict_RDD)

    predictions, rates_and_preds = predict()
