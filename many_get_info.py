from libbgg.apiv2 import BGG
from datetime import datetime
import re
from operator import itemgetter
from pymongo import MongoClient
import pickle
import numpy as np
import time


stat_results=[]
def get_stats_results():
    count = 1
    for current_call_lst in call_id_lst:
        time.sleep(np.random.choice(random_sec))
        print(count)
        print(current_call_lst[0], current_call_lst[-1])
        stat_results.append(conn.boardgame(current_call_lst, stats=True))
        count += 1
    return stat_results

def get_stats(id_game_dict, stat_results):
    count = 100
    for current_stat_results in stat_results:
        print(count)
        for current_game_stats in current_stat_results['items']['item']:
            # get id
            game_id = current_game_stats['id']
            # get game
            game = id_game_dict.get(game_id)
            #get image link
            try:
                image = current_game_stats['image']['TEXT']
            except:
                image = None
            #get description
            try:
                description = current_game_stats['description']['TEXT']
                #list of html tags that will be replaced by spaces
                replace_lst_dic = {"&#10;&#10;" : " ", "&#10;" : " ", "&rsquo;" : "'", "&ndash;" : "–", "&mdash" : "—"}
                for i, j in replace_lst_dic.items():
                    description = description.replace(i, j)
                description = re.sub(r'&ldquo;|&rdquo;|&quot;', '"', description)
                #get rid of white space
                description = " ".join(description.split())
            except:
                description = None
            #all link entries
            link = current_game_stats['link']
            #get board game categories
            categories = [(x['id'],x['value']) for x in link if x['type'] == 'boardgamecategory']
            if categories == []:
                categories = None
            #get board game mechanics
            mechanics = [(x['id'],x['value']) for x in link if x['type'] == 'boardgamemechanic']
            if mechanics == []:
                mechanics = None
            #kickstarter?
            if {'id': '8374', 'type': 'boardgamefamily', 'value': 'Crowdfunding: Kickstarter'} in link:
                kickstarter = 'yes'
            else: kickstarter = 'no'
            #get board game designer
            designer = [(x['id'],x['value']) for x in link if x['type'] == 'boardgamedesigner']
            #min and max players
            min_players = int(current_game_stats['minplayers']['value'])
            max_players = int(current_game_stats['maxplayers']['value'])
            #play times
            min_playtime = int(current_game_stats['minplaytime']['value'])
            playing_time = int(current_game_stats['playingtime']['value'])
            #min_age
            min_age = int(current_game_stats['minage']['value'])
            #suggestedplayers
            rec_players_poll = current_game_stats['poll'][0]
            #produces a list of tuples ((players, votes), ...)
            best_num_players_votes = []
            for players in rec_players_poll['results']:
                if all(var in players for var in ['numplayers', 'result']):
                    best_num_players_votes.append((players['numplayers'], int(players['result'][0]['numvotes'])))
                    #winner of best number of players
                    #if winner is n+, I am dropping the +
                    best_num_players = int((max(best_num_players_votes,key=itemgetter(1))[0])[0])
                else:
                    #no room for difference in dict structure...
                    best_num_players = None
            bnp_total_votes = int(rec_players_poll['totalvotes'])
            #stats dict
            stats_dict = current_game_stats['statistics']
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
            #I'm shelving this for now as I do not have immediate use for the
            #breakdown in ratings

            ranks = stats_dict['ratings']['ranks']['rank']
            if type(ranks) == list:
                rankings = []
                for x in ranks:
                    try:
                        rankings.append((x['friendlyname'],x['id'],int(x['value'])))
                    except:
                        rankings.append((x['friendlyname'],x['id'],None))
            else:
                if ranks['value'] == 'Not Ranked':
                    rankings = (ranks['friendlyname'],ranks['id'], None)
                else:
                    rankings = (ranks['friendlyname'],ranks['id'],int(ranks['value']))

            #number of ratings
            users_rated = int(stats_dict['ratings']['usersrated']['value'])
            #year published
            year_published = int(current_game_stats['yearpublished']['value'])

            # print("game:", game)
            # print("game_id:", game_id)
            # print("description:", description)
            # print("categories:", categories)
            # print("mechanics:" ,mechanics)
            # print("kickstarter:", kickstarter)
            # print("designer:", designer)
            # print("min_players:", min_players)
            # print("max_players:", max_players)
            # print("min_playtime:", min_playtime)
            # print("playing_time:", playing_time)
            # print("min_age:", min_age)
            # print("best_num_players:", best_num_players)
            # print("bnp_total_votes:", bnp_total_votes)
            # print("avg_rating:", avg_rating)
            # print("bayesavg_rating:", bayesavg_rating)
            # print("avg_weight:", avg_weight)
            # print("num_comments:", num_comments)
            # print("num_weights:", num_weights)
            # print("rankings:", rankings)
            # print("users_rated:", users_rated)
            # print("year_published:", year_published)
            # print("")
            # print("")

            ### to return all variables ###

            # return game, game_id, description, categories, mechanics, kickstarter, designer, min_players, min_playtime, playing_time, min_age, best_num_players, bnp_total_votes, avg_rating, bayesavg_rating, avg_weight, num_comments, num_weights, rankings, users_rated, year_published
            '''
            to insert all variables as a document into the specified collection
            '''
            # print(game_id)

            stats_to_mongo(stats_coll, game, game_id, description, categories, mechanics, kickstarter, designer, min_players, max_players, min_playtime, playing_time, min_age, best_num_players, bnp_total_votes, avg_rating, bayesavg_rating, avg_weight, num_comments, num_weights, rankings, users_rated, year_published)

        count += 100


def stats_to_mongo(stats_coll, game, game_id, description, categories, mechanics, kickstarter, designer, min_players, max_players, min_playtime, playing_time, min_age, best_num_players, bnp_total_votes, avg_rating, bayesavg_rating, avg_weight, num_comments, num_weights, rankings, users_rated, year_published):

        # category_bool = ",".join(['"{}_category": 1'.format(x[1]) for x in categories])


        stats_coll.update_one({"game_id": game_id }, {'$set' : {"game": game,
                        "game_id": game_id,
                        "description": description,
                        "categories": categories,
                        "mechanics": mechanics,
                        "kickstarter": kickstarter,
                        "designer": designer,
                        "min_players": min_players,
                        "max_players": max_players,
                        "min_playtime": min_playtime,
                        "playing_time": playing_time,
                        "min_age": min_age,
                        "best_num_players": best_num_players,
                        "bnp_total_votes": bnp_total_votes,
                        "avg_rating": avg_rating,
                        "bayesavg_rating": bayesavg_rating,
                        "avg_weight": avg_weight,
                        "num_comments": num_comments,
                        "num_weights": num_weights,
                        "users_rated": users_rated,
                        "year_published": year_published
                        }}, upsert=True)
        if categories:
            if type(categories) == list:
                for category in categories:
                    cat = '{}_category'.format(category[1])
                    stats_coll.update_one({'game_id' : game_id},
                         {'$set' : {cat : True}})
            else:
                cat = '{}_category'.format(categories[1])
                stats_coll.update_one({'game_id' : game_id},
                     {'$set' : {cat : True}})
        if mechanics:
            if type(mechanics) == list:
                for mechanic in mechanics:
                    mech = '{}_mechanic'.format(mechanic[1])
                    stats_coll.update_one({'game_id' : game_id},
                         {'$set' : {mech : True}})
            else:
                mech = '{}_mechanic'.format(mechanics[1])
                stats_coll.update_one({'game_id' : game_id},
                     {'$set' : {mech : True}})
        #if pulling rankings list into database
        if rankings:
            if type(rankings) == list:
                for ranking in rankings:
                    stats_coll.update_one({'game_id' : game_id},
                         {'$set' : {ranking[0]: int(ranking[2])}})
            else:
                stats_coll.update_one({'game_id' : game_id},
                     {'$set' : {rankings[0]: int(rankings[2])}})


if __name__ == '__main__':
    random_sec = np.random.uniform(5,7,[1000,])
    #open pickle file with ids,games,rating for use
    with open('data/official_game_dict_p2.pkl','rb') as fp:
        id_game_dict = pickle.load(fp)
    with open('data/game_ids_170516.pkl','rb') as fp:
        id_game_lst = pickle.load(fp)
    #make id,game key,value pair dictionary
    #was using a list here from get_ids.py, but now using updated dictionary
    #from get_official_names.py
    #id_game_dict = {x[0] : x[1] for x in id_game_lst[:-1]}

    client = MongoClient()
    #set database that collection is in
    database = client.bgg
    #collection for stats variables to go in
    stats_coll = database.game_stats
    #making call to api for dictionary object
    conn = BGG()
    '''
    the index in the id_game_lst specified here will determine how many games are looped through
    note an error is thrown if only one game is in the list
    one_get_info.py is for this purpose
    '''

    i_one = 0
    i_two = 100
    call_id_lst = []
    for r in range(145):
        call_id_lst.append([x[0] for x in id_game_lst[i_one:i_two]])
        i_one += 100
        i_two += 100

    stat_results = get_stats_results()
    get_stats(id_game_dict, stat_results)
