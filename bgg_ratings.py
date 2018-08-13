from libbgg.apiv2 import BGG
from datetime import datetime
import re
from operator import itemgetter
from pymongo import MongoClient
import pickle
import numpy as np
import time
from collections import defaultdict

'''
Inserts a record for each user's rating of a game in the specified collection with the following format:

    {"game": current_game,
    "game_id": bgg game id,
    "username": bgg username,
    "rating": users game rating,
    "comment": users comment on game
    }

Args:
    game_id_list_path (relative path):
        Filename path to pickle file created by `get_ids.py`
    error_path (pickle file path):
        Location that errors will be stored if any occur while calling the bgg api. Pickle file name needs to be included in path. It will be created if any errors occur.
    start_game (int): default is 0
        The index of the game in the game id list. When updating the database you would start with the first game. Only if an error occurs would it be necessary to change this variable
    start_page (int): default is 1
        The page of reviews of a game to start on. Again, when updating the database the default should be used. Each page has 100 reviews
    num_games (int): default is 1
        Number of games in the game id list to update. Default is 1000.
    database (mongo db database name): defaults to "bbg_test"
    collection (mongo db collection name): defaults to "game_comments_test"
'''

def insert_rating_record(game_id_list_path, error_path, start_game=0, start_page=1, num_games=1000, database="bgg_test", collection="game_comments_test"):
    error_lst = []

    conn = BGG()
    #open pickle file with ids,games,rating for use
    with open(game_id_list_path,'rb') as fp:
        id_game_lst = pickle.load(fp)

    client = MongoClient()
    #database for comments to go into
    database = "client.{}".format(database)
    #collection for stats variables to go in
    comments_coll = "database.{}".format(collection)
    print(comments_coll)

    id_game_dict = {x[0] : x[1] for x in id_game_lst[:-1]}
    #reverse lookup for dictionary
    # next(key for key, value in id_game_dict.items() if value == 'tzolk-mayan-calendar')

    #this range identifies the games that will be databased from the #id_game_lst
    call_id_lst = [x[start_game] for x in id_game_lst[:num_games]]
    print("Number of games:",len(call_id_lst))

    for game_id in call_id_lst:
        random_sec=np.random.uniform(5,6,[10000,])
        current_id = game_id
        current_game = id_game_dict.get(current_id)
        print("current_id:", current_id)
        print("current_game:", current_game)
        #specify starting page number, should be 1 unless testing
        page = start_page
        while page != None:
            time.sleep(np.random.choice(random_sec))
            print("page:", page)
            try:
                comment_results = conn.boardgame(game_id, comments=True, page=page, pagesize=100)
                #print(comment_results)
                try:
                    comments = comment_results['items']['item']['comments']['comment']
                    # print("comments:" ,comments)
                    print("length:",len(comments))
                    for entry in comments:
                        print(len(comments))
                        print(entry)
                        try:
                            rating = float(entry.get('rating'))
                        except ValueError:
                            rating = None
                        comments_coll.insert_one({"game": current_game,
                                        "game_id": str(current_id),
                                        "username": entry.get('username'),
                                        "rating": rating,
                                        "comment": entry.get('value')
                                            })
                    page += 1
                    #time.sleep(np.random.choice(random_sec))
                except KeyError:
                    #print("no comments")
                    page = None
            except KeyboardInterrupt:
                    raise
            except:
                error_lst.append((current_game,current_id,page,))
                #print(error_lst)
                with open(error_path, 'wb') as fp:
                    pickle.dump(error_lst, fp)
                page += 1
    return error_lst
