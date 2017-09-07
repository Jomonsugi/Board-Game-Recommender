import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import pickle

import seaborn as sns

# from pymongo import MongoClient
# client = MongoClient()
# db = client.bgg
# df = pd.DataFrame(list(db.game_comments.find()))
#
# def to_pickle(df):
#     with open('data/ratings_df.pkl', 'wb') as fp:
#         pickle.dump(df,fp)

def un_pickle():
    with open('data/ratings_df.pkl', 'rb') as fp:
        df = pickle.load(fp)
    return df

def ratings_plot(df_rated):
    df_rated = df_rated.dropna()
    users = df_rated['user_id'].value_counts()
    a = np.array(users)
    plt.hist(a, bins=50, color = 'blue')
    plt.xlim(0,200)
    plt.style.use('ggplot')
    plt.title('Number of Ratings', fontsize=50)
    plt.xlabel('Ratings')
    plt.ylabel('Users')
    # plt.rc('font', size=20)          # controls default text sizes
    plt.rc('axes', titlesize=20)     # fontsize of the axes title
    plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
    # plt.rc('legend', fontsize=20)    # legend fontsize
    plt.savefig('plots/ratings_200.png')

if __name__ == '__main__':
    df_rated = un_pickle()
    plt.close('all')
    ratings_plot(df_rated)
    # ratings_plot_seaborn(df_rated)
