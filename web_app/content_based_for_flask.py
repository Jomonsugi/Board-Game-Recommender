from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from pymongo import MongoClient
import pickle
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def get_data():
    client = MongoClient()
    db = client.bgg
    df = pd.DataFrame(list(db.game_stats.find()))
    return df

def mechanics(df):
    column_lst = list(df.columns.values)
    mechanics = [m for m in column_lst if 'mechanic' in m and m != 'mechanics']
    indexed_df = df.set_index(['game'])
    df_mechs = indexed_df[mechanics]
    for col in mechanics:
        df_mechs[col] = df_mechs[col] == True
    return df_mechs

def category(df):
    column_lst = list(df.columns.values)
    category = [c for c in column_lst if 'category' in c and c != 'categories']
    indexed_df = df.set_index(['game'])
    df_cats = indexed_df[category]
    for col in category:
        df_cats[col] = df_cats[col] == True
    return df_cats

def hamming(bool_df, df):
    distance_matrix = (1 - pairwise_distances(bool_df, metric = "hamming"))
    # top_10 = list(distance_matrix[0].argsort()[:-10:-1])
    # print(df.iloc[top_10,149])
    # print(distance_matrix)
    return distance_matrix

def jaccard(bool_df, df):
    distance_matrix = (1 - pairwise_distances(bool_df, metric = "jaccard"))
    # top_10 = list(distance_matrix[0].argsort()[:-10:-1])
    # print(df.iloc[top_10,149])
    return distance_matrix

def prep_columns(df):
    df = df[['Board Game Rank','game_id','game','description','playing_time','min_players', 'max_players', 'best_num_players', 'avg_rating', 'avg_weight']]
    df.columns = ['BGG Rank','game_id','Game','Description','Playing Time','Min Players', 'Max Players', 'Best Num Players', 'Avg Rating', 'Complexity']
    return df

def un_pickle_labeled_df():
    with open('../data/nmf_labeled_df_p2.pkl', 'rb') as fp:
        nmf_labeled_df = pickle.load(fp)
    return nmf_labeled_df

def for_flask_content(board_game, best_num_player, min_time, max_time):
    df = get_data()
    #using fuzzywuzzy to find best match
    choices = df['game'].tolist()
    board_game = process.extract(board_game, choices, limit=1)
    board_game = board_game[0][0]
    #creating df with only mechanic and category boolean values
    df_mechs = mechanics(df)
    df_cats = category(df)
    mech_cat_df = pd.concat([df_mechs, df_cats], axis=1)
    #calculating distance matrix
    distance_matrix = hamming(mech_cat_df, df)
    #finding index for board game and filtered list based on distance matrix
    idx = int((df.game[df.game == board_game].index.tolist())[0])
    sorted_idx = list(distance_matrix[idx].argsort()[::-1])
    sorted_df = df.iloc[sorted_idx,:]
    #changing column names for app output
    one_user_df = prep_columns(sorted_df)
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
    else:
        one_user_df = one_user_df.loc[(one_user_df['Best Num Players'].isin(best_num_player))]
        one_user_df = one_user_df.reset_index()
    one_user_df = one_user_df.round({'Complexity': 1 })
    #
    one_user_df['BGG Rank'] = one_user_df['BGG Rank'].fillna(0).astype(int)
    one_user_df['BGG Rank'] = one_user_df['BGG Rank'].astype(int)
    # rendered_df = one_user_df[['Game','Playing Time','Best Num Players' ,'Complexity']]
    rendered_df = one_user_df[['BGG Rank','Game','Playing Time','Min Players', 'Max Players','Best Num Players','Complexity']]
    rendered_df = rendered_df.loc[(rendered_df['BGG Rank'] < 3000) & rendered_df['BGG Rank'] != 0]
    return rendered_df.iloc[1:21,:]

if __name__ == '__main__':
    pass
    # rendered_df = for_flask_content('Terraforming Mars', 4, 0, 100)
