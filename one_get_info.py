from libbgg.apiv2 import BGG
from datetime import datetime
import re
from operator import itemgetter
from pymongo import MongoClient
import pickle

def get_stats_results():
    stat_results = conn.boardgame(5588, stats=True)
    return stat_results

def get_stats(id_game_dict, stat_results):
    # get id
    game_id = stat_results['items']['item']['id']
    # get game
    game = id_game_dict.get(int(game_id))
    #get image link
    image = stat_results['items']['item']['image']['TEXT']
    #get description
    description = stat_results['items']['item']['description']['TEXT']
    #list of html tags that will be replaced by spaces
    replace_lst_dic = {"&#10;&#10;" : " ", "&#10;" : " ", "&rsquo;" : "'"}
    for i, j in replace_lst_dic.items():
        description = description.replace(i, j)
    description = re.sub(r'&ldquo;|&rdquo;|&quot;', '"', description)
    #get rid of white space
    description = " ".join(description.split())
    #all link entries
    link = stat_results['items']['item']['link']
    #get board game categories
    categories = [(int(x['id']),x['value']) for x in link if x['type'] == 'boardgamecategory']
    if categories == []:
        categories = None
    #get board game mechanics
    mechanics = [(int(x['id']),x['value']) for x in link if x['type'] == 'boardgamemechanic']
    if mechanics == []:
        mechanics = None
    #kickstarter?
    if {'id': '8374', 'type': 'boardgamefamily', 'value': 'Crowdfunding: Kickstarter'} in link:
        kickstarter = 'yes'
    else: kickstarter = 'no'
    #get board game designer
    designer = [(int(x['id']),x['value']) for x in link if x['type'] == 'boardgamedesigner']
    #min and max players
    min_players = int(stat_results['items']['item']['minplayers']['value'])
    max_players = int(stat_results['items']['item']['maxplayers']['value'])
    #play times
    min_playtime = int(stat_results['items']['item']['minplaytime']['value'])
    playingtime = int(stat_results['items']['item']['playingtime']['value'])
    #min_age
    min_age = int(stat_results['items']['item']['minage']['value'])
    #suggestedplayers
    rec_players_poll = stat_results['items']['item']['poll'][0]
    #produces a list of tuples ((players, votes), ...)
    best_num_players_votes = []
    for players in rec_players_poll['results']:
        if all(var in players for var in ['numplayers', 'result']):
            best_num_players_votes.append((players['numplayers'], int(players['result'][0]['numvotes'])))
            #winner of best number of players
            best_num_players = max(best_num_players_votes,key=itemgetter(1))[0]
        else:
            #no room for difference in dict structure...
            best_num_players = [(None, None)]
    pnb_total_votes = int(rec_players_poll['totalvotes'])
    #stats dict
    stats_dict = stat_results['items']['item']['statistics']
    #average rating
    avg_rating = float(stats_dict['ratings']['average']['value'])
    #bayes average
    bayesavg_rating = float(stats_dict['ratings']['bayesaverage']['value'])
    #weight
    avg_weight = float(stats_dict['ratings']['averageweight']['value'])
    #number of comments
    num_comments = int(stats_dict['ratings']['numcomments']['value'])
    #number of weights
    num_weights = int(stats_dict['ratings']['numweights']['value'])
    #gives back a list of tuples with (ranking type,ranking id, ranking)
    ranks = stats_dict['ratings']['ranks']['rank']
    if type(ranks) == list:
        rankings = [(x['friendlyname'],x['id'],x['value']) for x in ranks]
    else:
        rankings = (ranks['friendlyname'],ranks['id'],ranks['value'])
    #number of ratings
    users_rated = int(stats_dict['ratings']['usersrated']['value'])
    #year published
    year_published = int(stat_results['items']['item']['yearpublished']['value'])

    print("game:", game)
    print("game_id:", game_id)
    print("description:", description)
    print("categories:", categories)
    print("mechanics:" ,mechanics)
    print("kickstarter:", kickstarter)
    print("designer:", designer)
    print("min_players:", min_players)
    print("max_players:", max_players)
    print("min_playtime:", min_playtime)
    print("playingtime:", playingtime)
    print("min_age:", min_age)
    print("best_num_players:", best_num_players)
    print("pnb_total_votes:", pnb_total_votes)
    print("avg_rating:", avg_rating)
    print("bayesavg_rating:", bayesavg_rating)
    print("avg_weight:", avg_weight)
    print("num_comments:", num_comments)
    print("num_weights:", num_weights)
    print("rankings:", rankings)
    print("users_rated:", users_rated)
    print("year_published:", year_published)

    ### to return all variables ###

    # return game, game_id, description, categories, mechanics, kickstarter, designer, min_players, min_playtime, playingtime, min_age, best_num_players, pnb_total_votes, avg_rating, bayesavg_rating, avg_weight, num_comments, num_weights, rankings, users_rated, year_published
    '''
    to insert all variables as a document into the specified collection
    '''
    # stats_to_mongo(stats_coll, game, game_id, description, categories, mechanics, kickstarter, designer, min_players, max_players, min_playtime, playingtime, min_age, best_num_players, pnb_total_votes, avg_rating, bayesavg_rating, avg_weight, num_comments, num_weights, rankings, users_rated, year_published)

def get_ratings_comments_results():
    pass
    # comment_results = conn.boardgame(169786 , ratingcomments=True, page=100, pagesize=100)
    # return comment_results

def get_ratings_comments(rc_dict):
    pass
    # comments = comment_results['items']['item']['comments']['comment']

def stats_to_mongo(stats_coll, game, game_id, description, categories, mechanics, kickstarter, designer, min_players, min_playtime, max_players, playingtime, min_age, best_num_players, pnb_total_votes, avg_rating, bayesavg_rating, avg_weight, num_comments, num_weights, rankings, users_rated, year_published):
        stats_coll.insert_one({"game": game,
                        "game_id": game_id,
                        "description": description,
                        "categories": categories,
                        "mechanics": mechanics,
                        "kickstarter": kickstarter,
                        "designer": designer,
                        "min_players": min_players,
                        "max_players": max_players,
                        "min_playtime": min_playtime,
                        "playingtime": playingtime,
                        "min_age": min_age,
                        "best_num_players": best_num_players,
                        "pnb_total_votes": pnb_total_votes,
                        "avg_rating": avg_rating,
                        "bayesavg_rating": bayesavg_rating,
                        "avg_weight": avg_weight,
                        "num_comments": num_comments,
                        "num_weights": num_weights,
                        "rankings": rankings,
                        "users_rated": users_rated,
                        "year_published": year_published
                        })

if __name__ == '__main__':
    #open pickle file with ids,games,rating for use
    with open('data/game_ids_170516.pkl','rb') as fp:
        id_game_lst = pickle.load(fp)
    #make id,game key,value pair dictionary
    id_game_dict = {x[0] : x[1] for x in id_game_lst[:-1]}

    client = MongoClient()
    database = client.bgg_test
    #collection for stats variables to go in
    stats_coll = database.game_stats
    #making call to api for dictionary object
    conn = BGG()
    stat_results = get_stats_results()
    stats_dict = stat_results['items']['item']
    get_stats(id_game_dict, stat_results)
