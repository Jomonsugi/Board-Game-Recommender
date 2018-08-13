from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
from pymongo import MongoClient
import pickle


def from_mongo():
    client = MongoClient()
    db = client.bgg
    username_lst = db.game_comments.distinct("username")
    return username_lst

def create_username_dictionary():
    username_dict = {}
    bad_id = []
    for username in username_lst[:1000]:
        print(str(username))
        try:
            url = "https://www.boardgamegeek.com/xmlapi2/user?name="+str(username)
            print(url)
            content = urlopen(url).read()
            soup = BeautifulSoup(content, "xml")

            body = str(soup.findAll('user'))
            user = body.split()[1]
            user_id = user.split('"')[1]
            username_dict[username] = user_id
        except:
            print("bad id", username)
            bad_id.append(username)
    return username_dict, bad_id

def to_pickle(username_dict):
    with open('data/username_dict.pickle', 'wb') as d:
        pickle.dump(username_dict, d)

def to_dic():
    counter = 1
    dic = {}
    for user_id in username_lst:
        dic[user_id] = counter
        counter += 1
    return dic

def update_db():
    client = MongoClient()
    db = client.bgg
    for k,v in id_dic.items():
        print(k,v)
        db.game_comments.update({ 'username': k } ,{ '$set' : { 'user_id': v } }, multi = True )

if __name__ == '__main__':
    username_lst = from_mongo()
    # username_dict, bad_id = create_username_dictionary()
    # to_pickle(username_dict)
    id_dic = to_dic()
    update_db()
