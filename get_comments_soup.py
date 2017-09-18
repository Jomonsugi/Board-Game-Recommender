from pymongo import MongoClient
import pickle
import numpy as np
import time
import requests
from bs4 import BeautifulSoup

def get_ratings_comments_results(ratings_coll):
    for game_id in call_id_lst:
        with open('user_dict_091617.pkl', 'rb') as fp:
            user_dict = pickle.load(fp)
        print("users in dictionary:", len(user_dict))
        # current_game = id_game_dict.get(current_id)
        print("current_id:", game_id)
        # print("current_game:", current_game)
        #specify starting page number, should be 1 unless testing
        page = 1
        while page != None:
            time.sleep(np.random.choice(random_sec))
            print("page:", page)
            r1 = requests.get("https://www.boardgamegeek.com/xmlapi2/thing?id="+str(game_id)+"&ratingcomments=1&page="+str(page)+"&pagesize=100")
            soup = BeautifulSoup(r1.text, "xml")
            # print(soup)
            name = soup.findAll("name")
            try:
                current_game = name[0]["value"]
            except:
                print("SOUP:")
                print(soup)
                print("NAME:")
                print(name)
            if current_game in mongo_games:
                print(current_game, " already in database")
                continue
            else:
                print(current_game)
                ratings = soup.findAll("comments")
                # l = comment.findAll('comment')
                for entry in ratings:
                    entry = entry
                tags = entry.findAll("comment")
                if tags:
                    for tag in tags:
                        try:
                            rating = tag['rating']
                            print(rating)
                            if len(rating) == 0:
                                rating = None
                        except:
                            rating = None
                        try:
                            username = tag['username']
                            print(username)
                            if len(username) == 0:
                                username == None
                        except:
                            username == None
                        try:
                            comment = tag['value']
                            print(comment)
                            if len(comment) == 0:
                                comment = None
                        except:
                            comment = None
                        if username in user_dict:
                            user_id = user_dict.get(username)
                            print("from dictionary:",user_id)
                        else:
                            try:
                                r2 = requests.get("https://www.boardgamegeek.com/xmlapi2/user?name="+username)
                                soup = BeautifulSoup(r2.text, "xml")
                                user_id = soup.findAll('user')[0]["id"]
                                user_dict[username]=user_id
                                print("from api:",user_id)
                                time.sleep(np.random.choice(random_sec))
                            except:
                                try:
                                    time.sleep(30)
                                    r2 = requests.get("https://www.boardgamegeek.com/xmlapi2/user?name="+username)
                                    soup = BeautifulSoup(r2.text, "xml")
                                    user_id = soup.findAll('user')[0]["id"]
                                    user_dict[username]=user_id
                                    time.sleep(np.random.choice(random_sec))
                                    print("from api:",user_id)
                                except:
                                    print("failed user id requests on:")
                                    print("page:", page)
                                    print("game:", current_game)
                                    raise ValueError('Error while calling API for user id')

                        ratings_coll.insert({"game": current_game,
                                        "game_id": str(game_id),
                                        "user_id" : str(user_id),
                                        "username": username,
                                        "rating": float(rating),
                                        "comment": comment})

                    page += 1
                    with open('user_dict_091617.pkl', 'wb') as fp:
                        pickle.dump(user_dict, fp)
                else:
                    with open('user_dict_091617.pkl', 'wb') as fp:
                        pickle.dump(user_dict, fp)
                    page = None


if __name__ == '__main__':
    random_sec = np.random.uniform(5,5.5,[10000,])
    '''
    This pickle file needs to be updated (update filename as well) using get_ids.py before running this script so that new games and updated ratings will be applied to the database
    '''
    with open('game_ids_091617.pkl','rb') as fp:
        id_game_lst = pickle.load(fp)

    client = MongoClient()
    #database for comments to go into
    database = client.bgg
    #collection for stats variables to go in
    ratings_coll = database.game_ratings
    mongo_games = ratings_coll.distinct("games")
    print("the current number of games in the database is", len(mongo_games))

    '''This range identifies the games that will be databased from the #id_game_lst. x[0] is the game id in id_game_lst. The index of id_game_lst will select that range of games that will be updated in the database'''
    call_id_lst = [x[0] for x in id_game_lst[995:1000]]
    get_ratings_comments_results(ratings_coll)
