import requests
from bs4 import BeautifulSoup
import re
import time
import numpy as np
import pickle

#each page has 100 games on it (other than the last page)
def get_game_ids(pages=1):
    url = 'https://www.boardgamegeek.com/browse/boardgame/page/'
    random_sec = np.random.uniform(5,7,[1000,])
    page_number=1
    id_list = []
    for page in range(pages):
        print("page number",page_number)
        req = requests.get(url+str(page_number))
        soup = BeautifulSoup(req.text, 'html.parser')
        thumbnails = soup.find_all('td',attrs={"class": "collection_thumbnail"})
        time.sleep(np.random.choice(random_sec))
        for thumbnail in thumbnails:
            game = thumbnail.a['href'].split('/')
            img = thumbnail.img['src'].split('/')
            start = 'pic'
            end = '_mt.jpg'
            img_id = ((img[4].split(start))[1].split(end)[0])
            id_list.append([str(game[2]),game[3], str(img_id)])
        page_number+=1
    return id_list

def small_image_links(id_list):
    small = {}
    for x in id_list:
        small[x[0]] = 'https://cf.geekdo-images.com/images/pic'+x[2]+'_t.jpg'
    return small

def medium_image_links(id_list):
    medium = {}
    for x in id_list:
        medium[x[0]] = 'https://cf.geekdo-images.com/images/pic'+x[2]+'_md.jpg'
    return medium

###replace current date in file name with date in format YYMMDD
def to_pickle_small(small):
    with open('data/small_images.pkl', 'wb') as fp:
        pickle.dump(small, fp)

def to_pickle_medium(medium):
    with open('data/medium_images.pkl', 'wb') as fp:
        pickle.dump(medium, fp)


if __name__ == '__main__':
    id_all = get_game_ids(20)
    small = small_image_links(id_all)
    medium = medium_image_links(id_all)
    to_pickle_small(small)
    to_pickle_medium(medium)
