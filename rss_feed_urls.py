import feedparser
import datetime
import json
from pathlib import Path
from load_data import load_urls
from newspaper import Article
from pymongo import MongoClient
import time
import subprocess
import socket


def add_to_mongo(tab, url, url_name):
    if already_exists(tab, url):
        return False
    try:
        a = Article(url)
        attempt = 0
        while a.html == '' and attempt < 10:
            a = Article(url)
            a.download()
            attempt += 1
        if attempt >= 10:
            return 'Article would not download!'
        if a.is_downloaded:
            a.parse()
        else:
            return 'Article would not download!'
    except:
        return 'Article would not download!'
    try:
        headline = a.title
    except:
        return 'No title!'
    try:
        date_published = a.publish_date
        if date_published == '' or date_published == None:
            date_published = datetime.datetime.now()
    except:
        date_published = datetime.datetime.now()
    try:
        author = a.authors
    except:
        author = None
    try:
        article_text = a.text
    except:
        return 'No text!'

    insert = {'url': url,
              'source': url_name,
              'headline': headline,
              'date_published': date_published,
              'author': author,
              'article_text': article_text}
    tab.insert_one(insert)
    return False


def already_exists(tab, url):
    return bool(tab.find({'url': url}).count())

urls = ['http://rss.cnn.com/rss/cnn_allpolitics.rss', 'http://feeds.abcnews.com/abcnews/politicsheadlines', 'http://feeds.foxnews.com/foxnews/politics', 'http://rss.nytimes.com/services/xml/rss/nyt/Politics.xml', 'http://hosted2.ap.org/atom/APDEFAULT/89ae8247abe8493fae24405546e9a1aa', 'http://feeds.reuters.com/Reuters/PoliticsNews', 'http://feeds.washingtonpost.com/rss/politics', 'https://www.prisonplanet.com/feed.rss', 'http://www.economist.com/sections/united-states/rss.xml', 'http://www.huffingtonpost.com/feeds/verticals/politics/index.xml', 'http://www.esquire.com/rss/news-politics.xml', 'http://www.rollingstone.com/politics/rss', 'http://www.cbsnews.com/latest/rss/politics', 'https://fivethirtyeight.com/politics/feed/', 'https://www.vox.com/rss/index.xml', 'http://feeds.feedburner.com/timeblogs/swampland', 'http://feeds.slate.com/slate-101526', 'http://www.washingtontimes.com/rss/headlines/news/politics/', 'http://feeds.feedburner.com/thedailybeast/politics']

url_names = ['cnn', 'abc', 'fox', 'nyt', 'ap', 'reuters', 'wapo', 'infowars', 'economist', 'huffpo', 'esquire', 'rollingstone', 'cbs', '538', 'vox', 'time', 'slate', 'washtimes', 'dailybeast']

while True:
    now = datetime.datetime.now()

    for url, url_name in zip(urls, url_names):
        print('Working on: ['+url_name+']. Added urls below:')
        socket.setdefaulttimeout(30)
        try:
            feed = feedparser.parse(url)
        except socket.timeout:
            print(url_name+' timed out!')
        links = set()
        for item in feed['items']:
            if url_name != 'vox':
                links.add(item['link'])
            elif 'politics' in item['link']:
                links.add(item['link'])

        client = MongoClient()
        db = client['rss_feeds_new']
        tab = db[url_name+'_'+now.strftime('%Y%m%d')]

        file_path = '../rss_url_files/{0}'.format(url_name+'_'+now.strftime('%Y%m%d'))
        my_file = Path(file_path)
        if my_file.is_file():
            old_urls = load_urls(file_path)
            for new_url in links:
                if new_url not in old_urls:
                    print(new_url)
                    result = add_to_mongo(tab, new_url, url_name)
            links.update(set(old_urls))
        else:
            for link in links:
                print(link)
                result = add_to_mongo(tab, link, url_name)

        f = open(file_path, 'w')
        f.write(json.dumps(list(links)))
        f.close()

    # Run script to upload datbase to S3 Bucket
    print('Backing up to S3 Bucket')
    p = subprocess.Popen('/home/ubuntu/backup.sh', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    print('Finished backing up to S3 Bucket')

    # sleep for an hour
    print('Sleeping for an hour...')
    time.sleep(60*60)
