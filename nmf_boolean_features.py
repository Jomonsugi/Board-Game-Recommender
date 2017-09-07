from sklearn.decomposition import NMF as NMF_sklearn
from pymongo import MongoClient
import pickle
import numpy as np
import pandas as pd

def load_df():
    client = MongoClient()
    db = client.bgg
    df = pd.DataFrame(list(db.game_stats.find()))
    return df

def make_cat_matrix(df,column_lst):
    categories = [c for c in column_lst if 'category' in c and c != 'category']
    indexed_df = df.set_index(['game'])
    df_categories = indexed_df[categories]
    for col in categories:
        df_categories[col] = df_categories[col] == True
    df_categories = df_categories.transpose()
    return df_categories, df_categories.as_matrix()

def make_mech_matrix(df,column_lst):
    mechanics = [m for m in column_lst if 'mechanic' in m and m != 'mechanics']
    indexed_df = df.set_index(['game'])
    df_mechs = indexed_df[mechanics]
    for col in mechanics:
        df_mechs[col] = df_mechs[col] == True
    return df_mechs, df_mechs.as_matrix()

def do_nmf(X):
    nmf = NMF_sklearn(n_components=30, max_iter=100, random_state=34, alpha=0.0, verbose = True)
    W = nmf.fit_transform(X)
    H = nmf.components_
    print('reconstruction error:', nmf.reconstruction_err_)
    return W, H

def topic_labels(H):
    for i, row in enumerate(cat_H):
        top_five = np.argsort(row)[::-1][:5]
        print('category_cluster', i)
        print('top_5', ' '.join([top_five[i] for i in x]))

def cluster(cluster, cat_H, df_categories):
    clusters = [np.argmax(cat_H[:,x]) for x in range(1000)]
    current_cluster = [i for i,x in enumerate(clusters) if x == cluster]
    games = list(df_categories.columns.values)
    games_in_cluster = [games[i] for i in current_cluster]
    return games_in_cluster


if __name__ == '__main__':
    df = load_df()
    column_lst = list(df.columns.values)
    df_categories, cat_mat = make_cat_matrix(df,column_lst)
    # # df_mechs, mech_mat = make_mech_matrix(df,column_lst)
    cat_W, cat_H = do_nmf(cat_mat)
    # mech_W, mech_H = do_nmf(mech_mat)
    # topic_labels(cat_H)
    cluster_15 = (15, cat_H, df_categories)
