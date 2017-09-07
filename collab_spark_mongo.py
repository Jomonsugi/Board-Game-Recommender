import pyspark as ps
from pymongo import MongoClient
import numpy as np
import random
from pyspark import SparkConf, SparkContext
from pyspark.ml.recommendation import ALS
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


spark = ps.sql.SparkSession.builder \
        .master("local[*]") \
        .appName("collab_rec") \
        .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/bgg_test.game_comments_test") \
        .getOrCreate()

sc = spark.sparkContext
sc.setCheckpointDir('checkpoint/')
sqlContext = SQLContext(sc)

def mongo_to_rdd_df():
    df = spark.read.format("com.mongodb.spark.sql.DefaultSource").load()
    ugr_data = df.select('user_id','game_id','rating').repartition(16).cache()
    ugr_df = ugr_data.filter("rating is not null")
    ugr_rdd = ugr_df.rdd
    ugr_df = ugr_df.withColumn("game_id", ugr_df["game_id"].cast("int"))
    return ugr_df, ugr_rdd

def train_val_test_rdd(ugr_rdd):
    train, validation, test = ugr_rdd.randomSplit([6, 2, 2], seed=0L)
    validation_for_predict = validation.map(lambda x: (x[0], x[1]))
    test_for_predict = test.map(lambda x: (x[0], x[1]))
    return train, validation, test, validation_for_predict, test_for_predict

def predict_test_rdd(train, validation, test, validation_for_predict, test_for_predict):
    global bestRank
    global bestLambda
    global bestNumIter
    seed = 5L
    tolerance = 0.02
    min_error = float('inf')
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1
    ranks = [4]
    lambdas = [0.1]
    numIters = [10,20]
    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        model = ALS.train(train, rank, numIter, lmbda, seed=seed)
        predictions = model.predictAll(validation_for_predict).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
        print('rank: {}, lmbda: {}, numIter: {}, RMSE: {}'.format(rank, lmbda, numIter, error))
        if error < min_error:
            min_error = error
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    print 'The best model was trained with \n rank: {} \n lambda: {} \n bestNumIter: {} \n RMSE: {}'.format(bestRank, bestLambda, bestNumIter, min_error)
    return predictions, rates_and_preds

def train_val_test_df(ugr_df):
    (df_train, df_val, df_test) = ugr_df.randomSplit([0.6, 0.2, 0.2], seed=0L)
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
    ranks = [10]
    lambdas = [0.05]
    numIters = [30]
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
        print "RMSE (validation) = {} for the model trained with rank: {}, lambda: {}, numIter: {} ".format(validationRmse, rank, lmbda, numIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

    print('The best model was trained with \n rank: {} \n lambda: {} \n bestNumIter: {} \n RMSE: {}'.format(bestRank, bestLambda, bestNumIter, bestValidationRmse))
    optimized_model = bestModel
    return optimized_model

def rdd_predict(ugr_rdd):
    seed = 5L
    rank = 10
    iterations = 10
    lmbda = .01

    full_train, full_test = ugr_rdd.randomSplit([7, 3], seed=0L)

    optimized_model = ALS.train(full_train, rank = rank, seed=seed,
                           iterations=iterations, lambda_= lmbda)

    data_for_predict = full_test.map(lambda x: (x[0], x[1]))
    predictions = optimized_model.predictAll(data_for_predict).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = full_test.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())

    print 'RMSE for test data: {}'.format(error)
    return optimized_model

def df_predict(df_test, optimized_model):
    seed = 5L
    rank = 10
    numIter = 30
    lmbda = .05

    # this code is for a sanity check. I converted all ratings to 0.0 and then use by using the new_test_df to predict below, I could see that predictions are made disregarding the rating row, making the rmse valid
    # print(df_test.show(10))
    # name = 'rating'
    # udf = UserDefinedFunction(lambda x: 'new_value', DoubleType())
    # new_test_df = df_test.select(*[udf(column).alias(name) if column == name else column for column in df_test.columns])
    # new_test_df=new_test_df.na.fill(0.0)
    # print("new_test_df:", new_test_df.show(10))

    predictions = optimized_model.transform(df_test)
    # print(predictions.take(3))

    predictions_drop = predictions.dropna()
    # print(predictions_drop.take(3))
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions_drop)
    print(rmse)
    print 'RMSE for test data: {}'.format(rmse)

def predict_for_user_rdd(ugr_rdd, ugr_df, username="jomonsugi"):
    client = MongoClient()
    coll = client.bgg_test.game_comments_test
    try:
        idn = coll.find_one({'username':username})['user_id']
    except TypeError:
        print("username not found")
        return
    user_ugr = ugr_rdd.filter(lambda x: x[0] == idn)
    #now get games that the user has not rated
    rated_game_ids_lst = [x["game_id"] for x in list(coll.find({"username": "ahalm"}))]

    # not_rated_rdd_test = ugr_rdd.filter(lambda: x[0] != 19)
    # print(type(not_rated_rdd_test))
    print(rated_game_ids_lst)
    # print("rated_game_ids_lst type:", type(rated_game_ids_lst))
    user_unrated_games = ugr_rdd.filter(lambda x: x[1] not in rated_game_ids_lst).map(lambda x: (idn, x[1], x[2]))
    # print("user_unrated_games type:" ,type(user_unrated_games))
    # print("user_unrated_games:", user_unrated_games.take(2))

    # lst = [(174430, 3.4), (169786, 3.4)]
    # rdd = sc.parallelize(lst)
    user_unrated_games = ugr_rdd.filter(lambda x: x[1] not in rated_game_ids_lst).map(lambda x: (idn, int(x[1]), x[2]))
    print(user_unrated_games.collect())
    print(type(user_unrated_games))

    schema = StructType( [
        StructField('user_id', IntegerType()),
        StructField('game_id', IntegerType()),
        StructField('rating', DoubleType())
        ])

    df_test = spark.createDataFrame(user_unrated_games, schema)
    return df_test

    # df = sqlContext.createDataFrame(user_unrated_games)
    # df = sqlContext.createDataFrame(user_unrated_games, ['user_id', 'game_id', 'rating'])
    # print(test.show(1))
    # print(type(user_unrated_games))
    # print("un-rated games by {}: {}".format(username, user_unrated_games.count()))
    # new_user_recommendations = optimized_model.predictAll(user_unrated_games)
    # print("new_user_recommendations:")
    # print(new_user_recommendations.collect())

def predict_for_user_df(ugr_df, optimized_model, username="jomonsugi"):
    client = MongoClient()
    coll = client.bgg_test.game_comments_test
    try:
        idn = coll.find_one({'username':username})['user_id']
    except TypeError:
        print("username not found")
        return
    #list of games the user has rated
    rated_game_ids_lst = [x["game_id"] for x in list(coll.find({"username": "ahalm"}))]
    #make a df with all games user hasn't rated, only with his user_id
    user_df = ugr_df.filter(ugr_df.user_id != idn)
    print(user_df.collect())

def predicted_to_df():
    pass


if __name__ == '__main__':
    #-------------------------------------------
    #note that mllib must be used on functions that use the rdd pipeline when #using the ALS library and ml for the DF pipeline
    #-------------------------------------------
    ugr_df, ugr_rdd = mongo_to_rdd_df()

    #---------------------RDD Pipeline----------------------
    #### for optimizing hyperparemeters ON PARTIAL DATA ####
    # train, validation, test, validation_for_predict, test_for_predict = train_val_test_rdd(ugr_rdd)
    #
    # predictions, rates_and_preds = predict_test_rdd(train, validation, test, validation_for_predict, test_for_predict)
    # optimized_model = rdd_predict(ugr_rdd)
    df_test = predict_for_user_rdd(ugr_rdd, ugr_df, username="ahalm")

    #---------------------DF Pipeline----------------------
    # df_train, df_val, df_test = train_val_test_df(ugr_df)
    # evaluator = make_evaluator()
    # optimized_model = predict_test_df(df_train, df_val, evaluator)
    # df_predict(ugr_df, optimized_model)
    # predict_for_user_df(ugr_df, optimized_model, username="ahalm")
