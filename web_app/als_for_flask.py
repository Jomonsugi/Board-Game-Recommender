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
        .config("spark.mongodb.input.uri", "mongodb://127.0.0.1/bgg.game_comments") \
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
    print 'RMSE for test data: {}'.format(rmse)
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

def un_pickle_labeled_df():
    with open('../data/nmf_labeled_df_p2.pkl', 'rb') as fp:
        nmf_labeled_df = pickle.load(fp)
    return nmf_labeled_df

def pickle_one_user_df(one_user_df):
    with open('../data/one_user_df_p2.pkl', 'wb') as fp:
        pickle.dump(one_user_df,fp, protocol=2)

def one_user_to_pd(nmf_labeled_df, one_user_predictions):
    nmf_labeled_df['game_id'] = nmf_labeled_df['game_id'].astype(int)
    one_user_df = one_user_predictions.toPandas()
    one_user_df = one_user_df.drop('rating', 1)
    pickle_one_user_df(one_user_df)
    merged = pd.merge(nmf_labeled_df,one_user_df,on=['game_id','game_id'])
    merged = merged.sort(columns='prediction', ascending=False)
    return merged

#putting at least one game from each top in the top 10
def redistribute(one_user_df):
    idx = one_user_df.index.values.tolist()
    nmf = list(one_user_df['Topic'])
    pairs = tuple(zip(idx,nmf))
    topics = [1,2,3,4,5,6,7,8]
    nmf_8_idx = []
    for pair in pairs:
        if pair[1] in topics:
            nmf_8_idx.append(pair[0])
            topics.remove(pair[1])
    print("indexes:", nmf_8_idx)
    nmf_top_10 = one_user_df.iloc[nmf_8_idx]
    dropped_df = one_user_df.drop(one_user_df.index[nmf_8_idx])
    nmeffed_df = nmf_top_10.append(dropped_df)
    return nmeffed_df

def prep_columns(df):
    df = df[['Board Game Rank','game_id','game','description','playing_time','min_players', 'max_players', 'best_num_players', 'avg_rating', 'avg_weight', 'nmf', 'Game', 'prediction']]
    df.columns = ['BGG Rank','game_id','game','Description','Playing Time','Min Players', 'Max Players', 'Best Num Players', 'Avg Rating', 'Complexity', 'Topic', 'Game', 'Prediction']
    return df

#once function rules them all
def for_flask(user_id, best_num_player, min_time, max_time):
    nmf_labeled_df = un_pickle_labeled_df()
    ugr_df, ugr_rdd = mongo_to_rdd_df()
    optimized_model = ALSModel.load("/Users/micahshanks/Galvanize/capstone/data/als_model")
    user_unrated_df = to_user_unrated_df(ugr_rdd, ugr_df, username=user_id)
    one_user_predictions = predict_one_user(user_unrated_df, optimized_model)
    one_user_df = one_user_to_pd(nmf_labeled_df, one_user_predictions)
    one_user_df = prep_columns(one_user_df)
    print(one_user_df.head())
    #for minimum and maximum time
    if min_time:
        min_time = int(min_time)
        one_user_df = one_user_df.loc[one_user_df['Playing Time'] > min_time]
    if max_time:
        max_time = int(max_time)
        one_user_df = one_user_df.loc[one_user_df['Playing Time'] < max_time]
        one_user_df = one_user_df.reset_index()
    #for best number of players
    if 'Any' in best_num_player or best_num_player == []:
        best_num_player = [1,2,3,4,5]
    best_num_player = [int(x) for x in best_num_player]
    if 5 in best_num_player:
        one_user_df = one_user_df.loc[(one_user_df['Best Num Players'] > 5) | one_user_df['Best Num Players'].isin(best_num_player)]
        one_user_df = one_user_df.reset_index()
        print(one_user_df.head())
    else:
        one_user_df = one_user_df.loc[(one_user_df['Best Num Players'].isin(best_num_player))]
        one_user_df = one_user_df.reset_index()
        print(one_user_df.head())
    #now we are ready to redistribute the top 8 based on topics
    one_user_df = redistribute(one_user_df)
    one_user_df = one_user_df.round({'Prediction': 1, 'Complexity': 1 })
    rendered_df = one_user_df[['BGG Rank','Game','Playing Time', 'Min Players', 'Max Players', 'Best Num Players' ,'Complexity']]
    # rendered_df = one_user_df[['BGG Rank','Game','Prediction', 'Topic']]
    return rendered_df.iloc[0:20,:]
