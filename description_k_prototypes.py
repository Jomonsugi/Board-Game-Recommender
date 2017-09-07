#!/usr/bin/env python

import numpy as np
from kmodes import kmodes
from kmodes import kprototypes
import pandas as pd
from pymongo import MongoClient


def kproto(syms, X , n_clusters = 10, init = 'Cao'):
    kproto = kprototypes.KPrototypes(n_clusters=10, init='Cao', verbose=1)
    clusters = kproto.fit_predict(X, categorical=[1])
    #
    # Print cluster centroids of the trained model.
    print(kproto.cluster_centroids_)
    # Print training statistics
    print(kproto.cost_)
    print(kproto.n_iter_)
    #
    for s, c in zip(syms, clusters):
        print("Symbol: {}, cluster:{}".format(s, c))

def kmode(y,x):
    kmodes_huang = kmodes.KModes(n_clusters=10, init='Huang', verbose=1)
    kmodes_huang.fit(x)
    # Print cluster centroids of the trained model.
    print('k-modes (Huang) centroids:')
    print(kmodes_huang.cluster_centroids_)
    # Print training statistics
    print('Final training cost: {}'.format(kmodes_huang.cost_))
    print('Training iterations: {}'.format(kmodes_huang.n_iter_))

    # kmodes_cao = kmodes.KModes(n_clusters=10, init='Cao', verbose=1)
    # kmodes_cao.fit(x)
    #
    # # Print cluster centroids of the trained model.
    # print('k-modes (Cao) centroids:')
    # print(kmodes_cao.cluster_centroids_)
    # # Print training statistics
    # print('Final training cost: {}'.format(kmodes_cao.cost_))
    # print('Training iterations: {}'.format(kmodes_cao.n_iter_))

    # print('Results tables:')
    # for result in (kmodes_huang, kmodes_cao):
    #     classtable = np.zeros((4, 4), dtype=int)
    #     for ii, _ in enumerate(y):
    #         classtable[int(y[ii][-1]) - 1, result.labels_[ii]] += 1
    #
    #     print("\n")
    #     print("    | Cl. 1 | Cl. 2 | Cl. 3 | Cl. 4 |")
    #     print("----|-------|-------|-------|-------|")
    #     for ii in range(4):
    #         prargs = tuple([ii + 1] + list(classtable[ii, :]))
    #         print(" D{0} |    {1:>2} |    {2:>2} |    {3:>2} |    {4:>2} |".format(*prargs))

if __name__ == '__main__':
    client = MongoClient()
    db = client.bgg
    df = pd.DataFrame(list(db.game_stats.find()))

    columns = ['avg_weight','best_num_players','playing_time']
    mechanics = ['Acting_mechanic',
     'Action / Movement Programming_mechanic',
     'Action Point Allowance System_mechanic',
     'Area Control / Area Influence_mechanic',
     'Area Enclosure_mechanic',
     'Area Movement_mechanic',
     'Area-Impulse_mechanic',
     'Auction/Bidding_mechanic',
     'Betting/Wagering_mechanic',
     'Campaign / Battle Card Driven_mechanic',
     'Card Drafting_mechanic',
     'Chit-Pull System_mechanic',
     'Co-operative Play_mechanic',
     'Commodity Speculation_mechanic',
     'Crayon Rail System_mechanic',
     'Deck / Pool Building_mechanic',
     'Dice Rolling_mechanic',
     'Grid Movement_mechanic',
     'Hand Management_mechanic',
     'Hex-and-Counter_mechanic',
     'Line Drawing_mechanic',
     'Memory_mechanic',
     'Modular Board_mechanic',
     'Paper-and-Pencil_mechanic',
     'Partnerships_mechanic',
     'Pattern Building_mechanic',
     'Pattern Recognition_mechanic',
     'Pick-up and Deliver_mechanic',
     'Player Elimination_mechanic',
     'Point to Point Movement_mechanic',
     'Press Your Luck_mechanic',
     'Rock-Paper-Scissors_mechanic',
     'Role Playing_mechanic',
     'Roll / Spin and Move_mechanic',
     'Route/Network Building_mechanic',
     'Secret Unit Deployment_mechanic',
     'Set Collection_mechanic',
     'Simulation_mechanic',
     'Simultaneous Action Selection_mechanic',
     'Singing_mechanic',
     'Stock Holding_mechanic',
     'Storytelling_mechanic',
     'Take That_mechanic',
     'Tile Placement_mechanic',
     'Time Track_mechanic',
     'Trading_mechanic',
     'Trick-taking_mechanic',
     'Variable Phase Order_mechanic',
     'Variable Player Powers_mechanic',
     'Voting_mechanic',
     'Worker Placement_mechanic']


    games = pd.DataFrame.as_matrix(df['game'])

    prot_X = pd.DataFrame.as_matrix(df, columns = columns)
    kproto(games, prot_X, n_clusters = 8)

    # kmodes_X = pd.DataFrame.as_matrix(df, columns = mechanics)
    # kmodes_X[:, 0:50] = kmodes_X[:, 0:50].astype(str)
    # kmode(games, kmodes_X)
