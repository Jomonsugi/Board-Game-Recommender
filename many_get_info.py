from libbgg.apiv2 import BGG
from datetime import datetime
import re
from operator import itemgetter
from pymongo import MongoClient
import pickle
import numpy as np
import time
import math


class BGGInfoDicts(object):

    def __init__(self, game_id_data_path, n_games):
        self.game_id_data_path = game_id_data_path
        self.n_games = n_games

        with open(self.game_id_data_path,'rb') as fp:
            self.id_game_lst = pickle.load(fp)

    def get_id_game_dict(self):
        id_game_dict = {x[0]: x[1] for x in self.id_game_lst}
        return id_game_dict

    # return a list of libbgg.infodict.InfoDict
    def get_info_dicts(self):
        call_id_lst = []

        i_one = 0
        i_two = self.n_games if self.n_games < 100 else 100

        for r in range(math.ceil(self.n_games/100)):
            call_id_lst.append([x[0] for x in self.id_game_lst[i_one:i_two]])
            i_one += 100
            i_two += 100

        random_sec = np.random.uniform(5,6,[1000,])

        info_dicts=[]
        for current_call_lst in call_id_lst:
            if len(call_id_lst) > 1:
                time.sleep(np.random.choice(random_sec))
            info_dicts.append(BGG().boardgame(current_call_lst, stats=True))
        return info_dicts

def get_stats(id_game_dict, stat_results):
    count = 2
    print("length of stat_results", len(stat_results))
    for current_stat_results in stat_results:
        print(count)
        for current_game_stats in current_stat_results['items']['item']:
            print("THIS IS A NEW ONE")
            print(type(current_game_stats))
            print("")
            print("")
            print("")
            print(current_game_stats)
            # get id
            game_id = current_game_stats['id']
            print(type(game_id))
            print("this is the game id", game_id)
            print("here is the id_game_dict", id_game_dict)
            # get game
            game = id_game_dict.get(int(game_id))
            print("this is the game extracted", game)
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

            print("")
            print("")
            print("HERE IS THE METADATA WE GOT")
            print("game:", game)
            print("game_id:", game_id)
            # print("description:", description)
            print("categories:", categories)
            print("mechanics:" , mechanics)
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

            # stats_to_mongo(stats_coll, game, game_id, description, categories, mechanics, kickstarter, designer, min_players, max_players, min_playtime, playing_time, min_age, best_num_players, bnp_total_votes, avg_rating, bayesavg_rating, avg_weight, num_comments, num_weights, rankings, users_rated, year_published)

        count += 2
        return id_game_dict

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


    '''
    the index in the id_game_lst specified here will determine how many games are looped through
    note an error is thrown if only one game is in the list
    one_get_info.py is for this purpose
    '''

    """
    Here, we take a game list loaded from the latest pull of get_ids.py
    When making API calls, a list of 100 game ids max can be called. Thus, we make a list of list, each list containing 100 game ids. `get_stats_results` then returns a list of list, each list containing a 'libbgg.infodict.InfoDict' in place of each id given. `get_stats` extracts metadata from each InfoDict, assigning data to variables, which are then written into the mongo_db database.

    This process is burdensome and the code needs to be completely refactored. The if_name_main block should only contain calls to the functions, not variable assignment. The length of the id_game_lst should be checked and then API calls should be made based on that list length unless otherwise stated.
    """

    collection = MongoClient().database.collection
