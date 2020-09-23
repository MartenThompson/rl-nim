# Methods to conduct a single game of nim

from agents import QAgent
import sys
import random as rnd
import numpy as np


def play_nim(agent0, agent1, board):
    """
    Simulate a single complete game between two agents. 
    Mutate them appropriately during and after it ends.
    """
    
    if sum(board) <= 0:
        sys.exit('Illegal starting board')
    
    
    while True: 
        # board = agent0.move
        # if board == end:
            # agent0.learn_end(win=True)        # reward=1
            # agent1.learn_end(win=False)       # reward=-1
            # return/break 
        
        # board = agent1.move
        # if board == end:
            # agent0.learn_end(reward=-1)
            # agent1.learn_end(reward=1)
            # return/break
        
        # agent0.learn(board)                  # reward=0
        # agent1.learn(board)                  # reward=0
        
        
        
        print('hey')
        break



    

