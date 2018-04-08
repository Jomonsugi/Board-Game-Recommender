import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np
import datetime
import pickle

#each page has 100 games on it (other than the last page)
def get_game_ids(games=100, pickle=False):

    '''
    First step in updating the database.
    A list of list of games is returned and saved to a pickle in the following format:
        [game id, game name, current rating on bgg]
        ex. [174430, 'Gloomhaven', 1]
    Args:
    games (int) default 1000
        Must be a multiple of 100. If not, the next lowest number divisible by 100 is used. For example if 962 is passed, that will result in 900 games being returned in the list.
    '''
    if games < 100:
        print("Specified number of games must be a multiple of 100.")
    if games/100 != 1:
        print("Warning: You have specified a number of games that is not a multiple of 100. {} games will be added to list.".format(int(games/100)*100))

    url = 'https://www.boardgamegeek.com/browse/boardgame/page/'
    random_sec = np.random.uniform(5,7,[1000,])
    id_list = []
    for page in range(int(games/100)):
        req = requests.get(url+str(page+1))
        soup = BeautifulSoup(req.text, 'html.parser')
        thumbnails = soup.find_all('td',attrs={"class": "collection_thumbnail"})
        time.sleep(np.random.choice(random_sec))
        for thumbnail in thumbnails:
            current = thumbnail.a['href'].split('/')
            game = current[3].replace("-", " ")
            game = " ".join([x[:1].upper() + x[1:] for x in game.split()])
            id_list.append([int(current[2]),game])
        print("{} games".format((page+1)*100))
    for i,game in enumerate(id_list):
        game.append(i+1)
    if pickle is True:
        pkl_file_path = "data/game_ids/{}.pkl".format(datetime.date.today().strftime("%Y_%m_%d"))
        with open(pkl_file_path, 'wb') as fp:
            pickle.dump(id_list, fp)
    return id_list
