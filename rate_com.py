import pandas as pd
from pymongo import MongoClient
import pickle
from scipy.sparse import csr_matrix

def from_mongo():
    client = MongoClient()
    db = client.bgg
    df = pd.DataFrame(list(db.game_comments.find()))
    return df

def pickle_df(df):
    with open('data/rc_df.pkl', 'wb') as fp:
        pickle.dump(df,fp)

def load_df():
    with open('data/rc_df.pkl','rb') as fp:
        rc_df = pickle.load(fp)
        #drops any rows where rating is NaN
        rc_df = rc_df.dropna()
    return rc_df

def to_utility(df):
    username_u = sorted(list(df.username.unique()))
    game_id_u = sorted(list(df.game_id.unique()))

    data = df['rating'].tolist()
    row = df.username.astype('category', categories=username_u).cat.codes
    col = df.game_id.astype('category', categories=game_id_u).cat.codes
    sparse_matrix = csr_matrix((data, (row, col)), shape=(len(username_u), len(game_id_u)))

    td = sparse_matrix.todense()

    df = pd.DataFrame(td)

    return td

if __name__ == '__main__':
    df = from_mongo()
    pickle_df(df)
    rc = load_df()
    rc_utility = to_utility(rc)



#this will show counts of all values
#pd.value_counts(rc_utility.values.flatten())
