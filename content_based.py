from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from pymongo import MongoClient


def get_data():
    client = MongoClient()
    db = client.bgg
    df = pd.DataFrame(list(db.game_stats.find()))
    return df

def mechanics(df):
    column_lst = list(df.columns.values)
    # print(column_lst)
    mechanics = [m for m in column_lst if 'mechanic' in m and m != 'mechanics']
    indexed_df = df.set_index(['game'])
    df_mechs = indexed_df[mechanics]
    for col in mechanics:
        df_mechs[col] = df_mechs[col] == True
    return df_mechs

def category(df):
    column_lst = list(df.columns.values)
    # print(column_lst)
    category = [c for c in column_lst if 'category' in c and c != 'categories']
    indexed_df = df.set_index(['game'])
    df_cats = indexed_df[category]
    for col in category:
        df_cats[col] = df_cats[col] == True
    return df_cats

def hamming(bool_df, df):
    distance_matrix = (1 - pairwise_distances(bool_df, metric = "hamming"))
    top_10 = list(distance_matrix[0].argsort()[:-10:-1])
    print(df.iloc[top_10,149])
    # print(top_10)
    return distance_matrix

def jaccard(bool_df, df):
    distance_matrix = (1 - pairwise_distances(bool_df, metric = "jaccard"))
    top_10 = list(distance_matrix[0].argsort()[:-10:-1])
    print(df.iloc[top_10,149])
    # print(top_10)
    return distance_matrix


if __name__ == '__main__':
    df = get_data()
    df_mechs = mechanics(df)
    df_cats = category(df)
    # mechs_ham = hamming(df_mechs, df)
    # mechs_jac = jaccard(df_mechs, df)
    # cats_ham = hamming(df_cats, df)
    # cats_jac = jaccard(df_cats, df)

    mech_cat_df = pd.concat([df_mechs, df_cats], axis=1)
    cats_ham = hamming(mech_cat_df, df)
