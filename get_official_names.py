import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import pickle

#each page has 100 games on it (other than the last page)
def get_game_names(pages=1):
    url = 'https://www.boardgamegeek.com/browse/boardgame/page/'
    random_sec = np.random.uniform(5,7,[1000,])
    page_number=1
    official_name_dict = {}
    for page in range(pages):
        print("page number",page_number)
        req = requests.get(url+str(page_number))
        soup = BeautifulSoup(req.text, 'html.parser')
        thumbnails = soup.find_all('td',attrs={"class": "collection_objectname"})
        time.sleep(np.random.choice(random_sec))
        for thumbnail in thumbnails:
            game = thumbnail.a['href'].split('/')
            game_id = game[2]
            official = ((((str(thumbnail)).split('>'))[5]).split('<'))[0]
            official_name_dict[game_id] = official
        page_number+=1
    return official_name_dict

def replace(official):
    for k,v in official.items():
        v = v.replace('&amp;', '&')
        official[k] = v
    return official

def to_pickle(official):
    with open('data/official_game_dict_p2.pkl', 'wb') as fp:
        pickle.dump(official, fp, protocol=2)

if __name__ == '__main__':
    official = get_game_names(145)
    official = replace(official)
    to_pickle(official)
