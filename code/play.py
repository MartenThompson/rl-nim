# Primary script in project

from agents import QAgent
from agents import QtAgent
from agents import BayesAgent
from agents import PerfectAgent
from statesActions import get_hash
from statesActions import de_hash

import csv
from datetime import datetime
import numpy.random as rnd
import numpy as np
import pandas as pd
import sys


def play_nim(p1, p2, board_hash, win_reward, lose_reward):
    """
    Simulate a single complete game between two agents. 
    Mutate them appropriately during and after it ends.
    """
    
    if sum(de_hash(board_hash)) <= 0:
        sys.exit('Illegal starting board')
    
    turns = 0
    while True: 
        if 0 == turns:
            new_board_hash = p1.move(board_hash)
        else:
            new_board_hash = p1.move(new_board_hash)
            
       
        if 0 == sum(de_hash(new_board_hash)):
            # P1 wins
            p1.learn(1, win_reward)
            
            if 0 != turns:
                # can't teach it if it never got to move :(
                p2.learn(0, lose_reward) 
            
            break
        
        new_board_hash = p2.move(new_board_hash)
        
        if 0 == sum(de_hash(new_board_hash)):
            # P2 wins
            p1.learn(0, lose_reward)
            p2.learn(1, win_reward)
            break
        
        # no one has one yet
        p1.learn(None, 0)
        p2.learn(None, 0)

        turns += 1
        
    # game over


    

def train_agents(n_games, p1, p2, starting_board_hash, win_reward, lose_reward, trace):
    """
    Play n_games many games of Nim using starting_board_hash. Randomly choose
    who plays first with equal probability.
    """
    
    for i in range(n_games):
        if rnd.uniform(0,1,1)[0] > 0.5: 
            p1.started = True
            p2.started = False
            play_nim(p1, p2, starting_board_hash, win_reward, lose_reward)    
        else:
            p1.started = False
            p2.started = True
            play_nim(p2, p1, starting_board_hash, win_reward, lose_reward)
    

    
    if trace:
        p1.write()
        p2.write()
        
    return([[p1.optimal_per_start, p1.winlose_per_start], [p2.optimal_per_start, p2.winlose_per_start]])


