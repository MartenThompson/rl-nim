# Class definitions for agents

import numpy.random as rnd
import numpy as np
import sys
import numpy as np
from nimUtils import board_bitsum
from nimUtils import rand_move
from statesActions import initialize_Q
from statesActions import get_hash
from statesActions import de_hash

class QAgent():
    """
    off-policy, so moving is independent of learning
    """
    def __init__(self, num_piles, init_Qval, eps):
        self.games_played = 0
        self.games_won = 0
        self.Q = initialize_Q(num_piles, init_Qval)
        self.eps = eps

    def move(self, state_hash):
        # follow an epsilon greedy policy to make a move mid-game
        
        if rnd.uniform() < self.eps:
            # random move
            return rand_move(state_hash)
        else: 
            # greedy move
            acts = self.Q[state_hash]
            best_act = de_hash(max(acts, key=acts.get))
            state = de_hash(state_hash)
            new_state = [s - a for s,a in zip(state, best_act)]
            return get_hash(new_state)
            
        
        
        

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
        

