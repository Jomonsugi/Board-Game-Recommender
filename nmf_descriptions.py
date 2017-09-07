from sklearn.decomposition import NMF as NMF_sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import pickle
import numpy as np
import matplotlib.pylab as plt
from pymongo import MongoClient
import pandas as pd

STOP = ['player', 'game', 'piece', 'round', 'point', 'score']

def un_pickle_desc():
    with open('data/doc_lst.pkl', 'rb') as fp:
        desc_processed = pickle.load(fp)
    return desc_processed

def tfideffed(desc_processed):
    tfidfvect = TfidfVectorizer(max_df = .5, min_df = 1, stop_words=STOP)
    tfidf = tfidfvect.fit_transform(desc_processed)
    vocabulary = np.array(tfidfvect.get_feature_names())
    return tfidf, vocabulary

def do_nmf(V):
    nmf = NMF_sklearn(n_components=8, max_iter=100, random_state=34, alpha=.01, verbose = True)
    W = nmf.fit_transform(V)
    H = nmf.components_
    print('reconstruction error:', nmf.reconstruction_err_)
    return W, H

def do_nmf_loop(V):
    rec_err_lst = []
    for c in range(7,8):
        nmf = NMF_sklearn(n_components=c, max_iter=100, random_state=34, alpha=.01, verbose = True)
        W = nmf.fit_transform(V)
        H = nmf.components_
        print('reconstruction error:', nmf.reconstruction_err_)
        rec_err_lst.append(nmf.reconstruction_err_)
    return W, H, rec_err_lst

def topic_labels(H):
    hand_labels = []
    for i, row in enumerate(H):
        top_10 = np.argsort(row)[::-1][:20]
        print('topic', i+1)
        print('-->', ' '.join(vocabulary[top_10]))

def plot_bar(H, vocabulary):
    print("Plot highest weighted terms in basis vectors")
    for i, row in enumerate(H):
        top10 = np.argsort(row)[::-1][:10]
        val = np.take(row, top10)
        plt.figure(i+1)
        plt.barh(np.arange(10) + .5, val, color="blue", align="center")
        plt.yticks(np.arange(10) + .5, (vocabulary[top10]))
        plt.xlabel("Weight")
        plt.ylabel("Term")
        plt.title("Highest Weighted Terms in Basis Vector W%d" % (i + 1))
        plt.grid(True)
        plt.savefig("plots/documents_basisW%d.png" % (i + 1), bbox_inches="tight")


def rec_error_plot(rec_err_lst):
    import matplotlib.pyplot as plt
    plt.plot(rec_err_lst)
    # plt.ylabel('some numbers')
    plt.savefig("plots/nmf_rec_error.png")

def un_pickle_thumbs():
    with open('data/game_ids_170516.pkl', 'rb') as fp:
        thumbnails = pickle.load(fp)
    return thumbnails

def un_pickle_official():
    with open('data/official_game_dict_p2.pkl', 'rb') as fp:
        official = pickle.load(fp)
    return official

def label_games(W):
    client = MongoClient()
    db = client.bgg
    df = pd.DataFrame(list(db.game_stats.find()))
    df = df.ix[:999,:]
    category = [np.argmax(row) +1 for row in W]
    df['nmf'] = category
    official = un_pickle_official()
    official_df = pd.DataFrame(list(official.items()))
    official_df.columns = ['game_id','Game']
    df_id_lst = df['game_id'].tolist()
    official_df = official_df[official_df['game_id'].isin(df_id_lst)]
    df = pd.merge(df, official_df, on='game_id')
    df = df[['Board Game Rank','game_id','game', 'Game','description','nmf','playing_time','min_players', 'max_players', 'best_num_players', 'avg_rating', 'avg_weight']]
    df.columns = ['Board Game Rank','game_id','bgg_game', 'Game','Description','nmf','Playing Time','Min Players', 'Max Players', 'Best Num Players', 'avg_rating', 'Avg Weight']
    return df

def caps(df):
    games = df['Game'].tolist()
    pretty_games = []
    for game in games:
        game = game.replace("-", " ")
        game = " ".join([x[:1].upper() + x[1:] for x in game.split()])
        pretty_games.append(game)
    df['Game'] = pretty_games
    return df

def pickle_the_labeled_df(desc_df):
    with open('data/nmf_labeled_df.pkl', 'wb') as fp:
        pickle.dump(desc_df,fp)

def plot_cats(W):
    category = [np.argmax(row) for row in W]
    unique, counts = np.unique(category, return_counts=True)
    category_dict = dict(zip(unique, counts))
    n_groups = 8
    categories = [v for k,v in category_dict.items()]
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    plt.bar(index, categories, bar_width,
                     alpha=opacity,
                     color='r')
    plt.xlabel('Categories')
    plt.ylabel('Games')
    plt.title('Number of Games in Categories')
    plt.xticks(index, ('1', '2', '3', '4', '5', '6', '7', '8'))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # plt.close("all")
    desc_processed = un_pickle_desc()
    V, vocabulary = tfideffed(desc_processed)
    W, H = do_nmf(V)
    # # W, H, rec_err_lst = do_nmf_loop(V)
    topic_labels(H)
    # # rec_error_plot(rec_err_lst)
    # plot_bar(H, vocabulary)

    # thumbnails = un_pickle_thumbs()
    desc_df = label_games(W)
    # desc_df = caps(desc_df)
    pickle_the_labeled_df(desc_df)
