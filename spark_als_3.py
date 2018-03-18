from sys import argv
from pymongo import MongoClient
import numpy as np
import random
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import SQLContext, SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Transformer
import math
import itertools
from math import sqrt
from operator import add
import time
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.types import *
import pickle
import pandas as pd


spark = SparkSession \
        .builder \
        .master("local[*]") \
        .appName("collab_rec") \
        .getOrCreate()

sc = spark.sparkContext

sc.setCheckpointDir("board_game_recsys_directory")

def csv_to_rdd_df():
    ugr_df = spark.read.csv("/Users/micahshanks/Galvanize/capstone/data/mongo_csv/ratings_091617.csv",
                             header=True,
                             sep=",",
                             inferSchema=True)

    ugr_df = ugr_df.filter("rating is not null")
    # ugr_rdd = ugr_df.rdd
    ugr_df = ugr_df.withColumn("game_id", ugr_df["game_id"].cast("int"))
    # ugr_df.show(10)
    return ugr_df

def mongo_to_df():
    df = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("spark.mongodb.input.uri", "mongodb://127.0.0.1/bgg.game_comments").load()
    ugr_data = df.select('user_id','game_id','rating').repartition(16).cache()
    ugr_df = ugr_data.filter("rating is not null")
    # ugr_rdd = ugr_df.rdd
    ugr_df = ugr_df.withColumn("game_id", ugr_df["game_id"].cast("int"))
    return ugr_df

def train_val_test_df(ugr_df):
    (df_train, df_val, df_test) = ugr_df.randomSplit([0.6, 0.2, 0.2], seed=34)
    print("train test eval split done")
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
    lambdas = [0.75]
    numIters = [20]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1
    # df_train = df_train.na.drop()
    # df_val=df_val.na.drop()
    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        als = ALS(rank=rank, maxIter=numIter, regParam=lmbda, numUserBlocks=10, numItemBlocks=10, implicitPrefs=False,
                  alpha=5,
                  checkpointInterval=10,
                  userCol="user_id", itemCol="game_id", seed=1, ratingCol="rating")
        print(als)
        model=als.fit(df_train)
        print("still no problems here")
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

def to_user_unrated_df(ugr_df, username="jomonsugi"):
    client = MongoClient()
    coll = client.bgg.game_comments
    try:
        idn = coll.find_one({'username':username})['user_id']
    except TypeError:
        print("username not found")
        return

    user_rated_df = ugr_df.where(ugr_df.user_id == idn)

    game_id_df = ugr_df.select('game_id').distinct().withColumn('user_id', F.lit(idn)).withColumn('rating', F.lit(0.0))
    unrated_df = game_id_df.join(user_rated_df, 'game_id', 'left_anti').select(['user_id', 'game_id', 'rating'])

    #user_ugr = ugr_df.filter(lambda x: x[0] == idn)

    #now get games that the user has not rated
    # rated_game_ids_lst = [x["game_id"] for x in list(coll.find({"username": username}))]
    #print(rated_game_ids_lst)
    #create RDD where games are not rated which will include every entry
    #that is not a game rated by the user_id
    # user_unrated_games_df = ugr_rdd.filter(lambda x: x[1] not in rated_game_ids_lst).map(lambda x: (idn, int(x[1]), x[2]))
    # print(user_unrated_games_df)
    # #create data frame
    # schema = StructType( [
    #     StructField('user_id', IntegerType()),
    #     StructField('game_id', IntegerType()),
    #     StructField('rating', DoubleType())
    #     ])
    #
    # user_unrated_df = spark.createDataFrame(user_unrated_games_df, schema)
    # name = 'rating'
    # udf = UserDefinedFunction(lambda x: 'new_value', DoubleType())
    # new_test_df = user_unrated_df.select(*[udf(column).alias(name) if column == name else column for column in user_unrated_df.columns])
    # new_test_df=new_test_df.na.fill(0.0)
    # #drop all duplicates, thus creating a df with only games unrated by user
    # unique_games_df = new_test_df.dropDuplicates(['game_id'])
    # unique_games_df.show(10)
    return unrated_df

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
    with open('data/pickles/username_dict_p2.pickle', 'rb') as fp:
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

def un_pickle_labeled_df():
    with open('data/pickles/nmf_labeled_df.pkl', 'rb') as fp:
        nmf_labeled_df = pickle.load(fp)
    return nmf_labeled_df

def one_user_to_pd(nmf_labeled_df, one_user_predictions):
    one_user_df = one_user_predictions.toPandas()
    one_user_df = one_user_df.sort(columns='prediction', ascending=False)
    one_user_df = one_user_df.drop('rating', 1)
    frames = [one_user_df, nmf_labeled_df]
    one_user_df = pd.concat(frames)
    return one_user_df

if __name__ == '__main__':
    ugr_df = csv_to_rdd_df()
    # df_train, df_val, df_test = train_val_test_df(ugr_df)
    # evaluator = make_evaluator()
    # optimized_model = predict_test_df(df_train, df_val, evaluator)
    # optimized_model.save("/Users/micahshanks/Galvanize/capstone/data/als_model_test3")
    # df_predict_on_test(df_test, optimized_model)
    # optimized_model = ALSModel.load("/Users/micahshanks/Galvanize/capstone/data/als_model_test3")

    # # _, user = argv
    # # now on to one user predictions
    user_unrated_df = to_user_unrated_df(ugr_df, username='RomyCat')
    # one_user_predictions = predict_one_user(user_unrated_df, optimized_model)
    #
    # # and for all users to df
    # # note: this should not be done locally as it will take
    # # days
    # # all_users_recs_df = to_all_users_df(ugr_rdd, ugr_df)
    # nmf_labeled_df = un_pickle_labeled_df()
    # one_user_df = one_user_to_pd(nmf_labeled_df, one_user_predictions)
