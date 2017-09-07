from sys import argv
import pyspark as ps
from pymongo import MongoClient
import numpy as np
import random
from pyspark import SparkConf, SparkContext
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import SQLContext
from pyspark.sql.functions import col
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Transformer
import math
import itertools
from math import sqrt
from operator import add
import sys
import time
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import DoubleType
from pyspark.sql import Row
from pyspark.sql.types import *
import pickle
import pandas as pd


spark = ps.sql.SparkSession.builder \
        .master("local[*]") \
        .appName("collab_rec") \
        .getOrCreate()

def csv_to_rdd_df():
    ugr_df = spark.read.csv("/home/ubuntu/bgg-exploration/data/mongo_csv/ratings_091617.csv",
                             header=True,
                             sep=",",
                             inferSchema=True)

    ugr_df = ugr_df.filter("rating is not null")
    ugr_rdd = ugr_df.rdd
    ugr_df = ugr_df.withColumn("game_id", ugr_df["game_id"].cast("int"))
    return ugr_df, ugr_rdd

def mongo_to_rdd_df():
    df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
    ugr_data = df.select('user_id','game_id','rating').repartition(16).cache()
    ugr_df = ugr_data.filter("rating is not null")
    ugr_rdd = ugr_df.rdd
    ugr_df = ugr_df.withColumn("game_id", ugr_df["game_id"].cast("int"))
    return ugr_df, ugr_rdd

def train_val_test_df(ugr_df):
    (df_train, df_val, df_test) = ugr_df.randomSplit([0.6, 0.2, 0.2], seed=34)
    return df_train, df_val, df_test

def make_evaluator():
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    return evaluator

def computeRmse(model, data, evaluator):
    predictions = model.transform(data)
    predictions_drop = predictions.dropna()
    rmse = evaluator.evaluate(predictions_drop)
    print("Root-mean-square error = " + str(rmse))
    return rmse

def predict_test_df(df_train, df_val, evaluator):
    ranks = [5]
    lambdas = [0.2]
    numIters = [200]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1
    # df_train = df_train.na.drop()
    # df_val=df_val.na.drop()
    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        als = ALS(rank=rank, maxIter=numIter, regParam=lmbda, numUserBlocks=10, numItemBlocks=10, implicitPrefs=False,
                  alpha=1.0,
                  userCol="user_id", itemCol="game_id", seed=1, ratingCol="rating", nonnegative=True,
                  checkpointInterval=10, intermediateStorageLevel="MEMORY_AND_DISK", finalStorageLevel="MEMORY_AND_DISK")
        model=als.fit(df_train)

        validationRmse = computeRmse(model, df_val, evaluator)
        print("RMSE (validation) = {} for the model trained with rank: {}, lambda: {}, numIter: {} ".format(validationRmse, rank, lmbda, numIter))
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    print('The best model was trained with \n rank: {} \n lambda: {} \n bestNumIter: {} \n RMSE: {}'.format(bestRank, bestLambda, bestNumIter, bestValidationRmse))
    optimized_model = bestModel
    return optimized_model

def df_predict_on_test(df_test, optimized_model):
    #optimized hyperparemeters from predict_test_df
    seed = 1
    rank = 5
    numIter = 200
    lmbda = 0.2

    predictions = optimized_model.transform(df_test)
    print(predictions.count())
    print(predictions.take(3))
    predictions_drop = predictions.dropna()
    print(predictions_drop.count())
    print(predictions_drop.take(10))
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions_drop)
    print(rmse)
    print('RMSE for test data: {}'.format(rmse))
    return predictions

def to_user_unrated_df(ugr_rdd, ugr_df, username="jomonsugi"):
    client = MongoClient()
    coll = client.bgg.game_comments
    try:
        idn = coll.find_one({'username':username})['user_id']
    except TypeError:
        print("username not found")
        return
    user_ugr = ugr_rdd.filter(lambda x: x[0] == idn)
    #now get games that the user has not rated
    rated_game_ids_lst = [x["game_id"] for x in list(coll.find({"username": username}))]
    #create RDD where games are not rated which will include every entry
    #that is not a game rated by the user_id
    user_unrated_games_rdd = ugr_rdd.filter(lambda x: x[1] not in rated_game_ids_lst).map(lambda x: (idn, int(x[1]), x[2]))
    #create data frame
    schema = StructType( [
        StructField('user_id', IntegerType()),
        StructField('game_id', IntegerType()),
        StructField('rating', DoubleType())
        ])

    user_unrated_df = spark.createDataFrame(user_unrated_games_rdd, schema)
    name = 'rating'
    udf = UserDefinedFunction(lambda x: 'new_value', DoubleType())
    new_test_df = user_unrated_df.select(*[udf(column).alias(name) if column == name else column for column in user_unrated_df.columns])
    new_test_df=new_test_df.na.fill(0.0)
    #drop all duplicates, thus creating a df with only games unrated by user
    unique_games_df = new_test_df.dropDuplicates(['game_id'])
    return unique_games_df

def predict_one_user(user_unrated_df, optimized_model):
    #optimized hyperparemeters from predict_test_df
    seed = 1
    rank = 5
    numIter = 200
    lmbda = 0.2

    one_user_predictions = optimized_model.transform(user_unrated_df)
    print('TYPE:', type(one_user_predictions))
    sorted_predictions = one_user_predictions.sort("rating")
    return sorted_predictions

def un_pickle_user_dict():
    with open('data/username_dict_p2.pickle', 'rb') as fp:
        username_dict = pickle.load(fp)
    return username_dict

def to_all_users_df(ugr_rdd, ugr_df):
    client = MongoClient()
    coll = client.bgg.game_comments
    username_dict = un_pickle_user_dict()
    count = 1

    schema = StructType( [
        StructField('user_id', IntegerType()),
        StructField('game_id', IntegerType()),
        StructField('rating', DoubleType())
        ])

    all_users_recs_df = spark.createDataFrame(sc.emptyRDD(), schema)
    print(all_users_recs_df)
    for username, user_id in username_dict.items():
        if count > 10:
            break
        print("current iteration:", count)
        rated_game_ids_lst = [x["game_id"] for x in list(coll.find({"username": username}))]
        # create RDD where games are not rated which will include every entry that is not a game rated by the user_id
        user_unrated_games_rdd = ugr_rdd.filter(lambda x: x[1] not in rated_game_ids_lst).map(lambda x: (user_id, int(x[1]), x[2]))
        #create data frame
        user_unrated_df = spark.createDataFrame(user_unrated_games_rdd, schema)
        name = 'rating'
        udf = UserDefinedFunction(lambda x: 'new_value', DoubleType())
        new_test_df = user_unrated_df.select(*[udf(column).alias(name) if column == name else column for column in user_unrated_df.columns])
        new_test_df=new_test_df.na.fill(0.0)
        #drop the duplicates and at this point we have a unique df for the currenty user in iteration
        unique_games_df = new_test_df.dropDuplicates(['game_id'])
        # now iterate again and use .union on current data frame, repeat and return
        all_users_recs_df = all_users_recs_df.union(unique_games_df)
        print("rows in dataframe:", all_users_recs_df.count())
        count+=1
    return all_users_recs_df

def one_user_to_pd(one_user_predictions):
    one_user_df = one_user_predictions.toPandas()
    one_user_df = one_user_df.sort(columns='prediction', ascending=False)
    one_user_df = one_user_df.drop('rating', 1)
    return one_user_df

if __name__ == '__main__':
    ugr_df, ugr_rdd = csv_to_rdd_df()
    df_train, df_val, df_test = train_val_test_df(ugr_df)
    evaluator = make_evaluator()
    optimized_model = predict_test_df(df_train, df_val, evaluator)
    optimized_model.save("/home/ubuntu/bgg-exploration/data/als_model_new")
    df_predict_on_test(df_test, optimized_model)
    optimized_model = ALSModel.load("/home/ubuntu/bgg-exploration/data/als_model")

    # _, user = argv
    #now on to one user predictions
    # user_unrated_df = to_user_unrated_df(ugr_rdd, ugr_df, username='RomyCat')
    # one_user_predictions = predict_one_user(user_unrated_df, optimized_model)

    #and for all users to df
    #note: this should not be done locally as it will take
    #days
    # all_users_recs_df = to_all_users_df(ugr_rdd, ugr_df)
    one_user_df = one_user_to_pd(one_user_predictions)
