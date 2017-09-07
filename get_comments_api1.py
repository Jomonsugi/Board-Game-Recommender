from libbgg.apiv1 import BGG
from datetime import datetime
import re
from operator import itemgetter
from pymongo import MongoClient
import pickle
import numpy as np
import time


error_lst = []
def get_ratings_comments_results(call_id_lst, comments_coll):
    for game_id in call_id_lst:
        current_id = game_id
        current_game = id_game_dict.get(current_id)
        print("current_id:", current_id)
        print("current_game:", current_game)
        #specify starting page number, should be 1 unless testing
        page = 1
        while page != None:
            # time.sleep(np.random.choice(random_sec))
            print("page:", page)
            try:
                comment_results = conn.get_game(game_id, comments=True, comments_page=page)
                # print(comment_results)
                try:
                    comments = comment_results['boardgames']['boardgame']['comment']
                    # print("comments:" ,comments)
                    # print("length:",len(comments))
                    for entry in comments:
                        try:
                            rating = int(entry.get('rating'))
                        except ValueError:
                            rating = None
                        if entry.get('TEXT') == 'N/A':
                            continue
                        else:

                            comments_coll.insert_one({"game": current_game,
                                            "game_id": str(current_id),
                                            "username": entry.get('username'),
                                            "rating": rating,
                                            "comment": entry.get('TEXT')
                                                })
                    page += 1
                    time.sleep(np.random.choice(random_sec))
                except KeyError:
                    # print("no comments")
                    page = None
            except KeyboardInterrupt:
                    raise
            except:
                error_lst.append((current_game,page))
                print(error_lst)
                to_pickle(error_lst)
                page += 1
    return error_lst

def to_pickle(error_lst):
    with open('data/errors_api1.pkl', 'wb') as fp:
        pickle.dump(error_lst, fp)

if __name__ == '__main__':
    conn = BGG()
    random_sec = np.random.uniform(5,7,[10000,])
    #open pickle file with ids,games,rating for use
    with open('data/game_ids_170516.pkl','rb') as fp:
        id_game_lst = pickle.load(fp)

    client = MongoClient()
    #database for comments to go into
    database = client.bgg
    #collection for stats variables to go in
    comments_coll = database.game_comments

    id_game_dict = {x[0] : x[1] for x in id_game_lst[:-1]}
    #reverse lookup for dictionary
    # next(key for key, value in id_game_dict.items() if value == 'tzolk-mayan-calendar')

    #this range identifies the games that will be databased from the #id_game_lst
    # call_id_lst = [x[0] for x in id_game_lst[24:25]]
    call_id_lst = [193738,126163,103885,27833,4098]
    error_lst = get_ratings_comments_results(call_id_lst, comments_coll)

    '''
    great western trail, , 19378, 2
    [('tzolk-mayan-calendar', , 126163, 13),
     ('star-wars-x-wing-miniatures-game',103885,  31),
     ('steam', 27833, 14),
     ('age-steam', 4098,  16)]
     '''
