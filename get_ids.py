import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np
from datetime import datetime

#each page has 100 games on it (other than the last page)
def get_game_ids(pages=1):
    url = 'https://www.boardgamegeek.com/browse/boardgame/page/'
    random_sec = np.random.uniform(5,7,[1000,])
    page_number=1
    id_list = []
    for page in range(pages):
        req = requests.get(url+str(page_number))
        soup = BeautifulSoup(req.text, 'html.parser')
        thumbnails = soup.find_all('td',attrs={"class": "collection_thumbnail"})
        time.sleep(np.random.choice(random_sec))
        for thumbnail in thumbnails:
            current = thumbnail.a['href'].split('/')
            id_list.append([int(current[2]),current[3]])
        page_number+=1
        print("page number",page_number)
    for i,game in enumerate(id_list):
        game.append(i+1)
    id_list.append(datetime.now())
    return id_list

###replace current date in file name with date in format YYMMDD
def to_pickle(id_all):
    with open('data/game_ids_current_date.pkl', 'wb') as fp:
        pickle.dump(id_all, fp)

if __name__ == '__main__':
    id_all = get_game_ids(1)
    # to_pickle(id_all)
