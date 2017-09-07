from sklearn.decomposition import NMF as NMF_sklearn
from pymongo import MongoClient
import numpy as np
import pandas as pd
import nimfa

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

def do_nmf(V):
    nmf = nimfa.Nmf(V, seed='random_vcol', rank=10, max_iter=100)
    nmf_fit = nmf()

    W = nmf_fit.basis()
    print('Basis matrix:\n%s' % W)

    H = nmf_fit.coef()
    print('Mixture matrix:\n%s' % H)

    print("starting")
    r = nmf.estimate_rank(rank_range=[10, 15, 20, 25], what='all')
    # pp_r = '\n'.join('%d: %5.3f' % (rank, vals['all']) for rank, vals in r.items())
    print('Rank estimate:\n%s' % r)

    # print('Rss: %5.4f' % nmf_fit.fit.rss())
    # print('Rss: %5.4f' % nmf_fit.fit.coph_cor())
    # print('Evar: %5.4f' % nmf_fit.fit.evar())
    # print('K-L divergence: %5.4f' % nmf_fit.distance(metric='kl'))
    # print('Sparseness, W: %5.4f, H: %5.4f' % nmf_fit.fit.sparseness())


if __name__ == '__main__':
    df = load_df()
    column_lst = list(df.columns.values)
    df_categories, cat_mat = make_cat_matrix(df,column_lst)
    # # df_mechs, mech_mat = make_mech_matrix(df,column_lst)
    # do_nmf(cat_mat)
    do_nmf(cat_mat)
