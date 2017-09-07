import pandas as pd
import numpy as np
import pickle

def un_pickle_labeled_df():
    with open('data/nmf_labeled_df_p2.pkl', 'rb') as fp:
        nmf_labeled_df = pickle.load(fp)
    return nmf_labeled_df


def redistribute(nmf_labeled_df):
    idx = nmf_labeled_df.index.values.tolist()
    nmf = list(nmf_labeled_df['nmf'])
    pairs = tuple(zip(idx,nmf))
    topics = [1,2,3,4,5,6,7,8]
    nmf_10_idx = []
    for pair in pairs:
        if pair[1] in topics:
            nmf_10_idx.append(pair[0])
            topics.remove(pair[1])
    nmf_top_10 = nmf_labeled_df.iloc[nmf_10_idx]
    dropped_df = nmf_labeled_df.drop(nmf_labeled_df.index[nmf_10_idx])
    nmeffed_df = nmf_top_10.append(dropped_df)
    return nmf_top_10, nmeffed_df



if __name__ == '__main__':
    nmf_labeled_df = un_pickle_labeled_df()
    nmf_top_10, nmeffed_df = redistribute(nmf_labeled_df)
