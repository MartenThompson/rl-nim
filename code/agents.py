# Class definitions for agents

import random as rnd
import numpy as np
import sys
import numpy as np
from nimUtils import board_bitsum
from nimUtils import rand_move

class QAgent():
    """
    off-policy, so moving is distinct from learning
    """
    def __init__(self):
        self.games_played = 0
        self.games_won = 0
        self.Q = np.zeros([2,2])

    def move(board):
        # make a move mid-game
        print('move method')
        
        

    def learn():
        # learning mid-game
        print('learn method')
    
    def learn_end():
        # learning after game ends
        print('learn_end method')





class PerfectAgent():
    def move(self, board): 
        # make an omptimal move mid-game
        # mutates board by reference
        
                
        # optimal moves are 
            # 1: taking the final pile to win
            # 2: reducing the board's bitsum to 0 when possible
        
        if sum(board) <= 0:
            sys.exit('Game already over or in illegal state')

        
        # 1
        if np.count_nonzero(board) == 1: 
            board[:] = np.zeros(len(board), dtype=int)      #.tolist()
            return()
                
        # 2
        for i in range(len(board)):
            for j in (np.arange(board[i]) + 1):     # +1 because can't take 0
                temp_board = board[:]
                temp_board[i] -= j                  # take j items from pile i
                
                if board_bitsum(temp_board) == 0:
                    board[i] -= j
                    return()
                
        
        # no optimal move possible, move randomly
        rand_move(board)
        

