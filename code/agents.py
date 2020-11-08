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
    def __init__(self, num_piles, starting_board, init_Qval, eps):
        self.games_played = 0
        self.games_won = 0
        # call Q(state) to get available actions. Call avail_acts(act_hash) to get value
        self.Q = initialize_Q(num_piles, init_Qval)
        self.eps = eps
        self.alpha = 0.1
        self.gamma = 1
        self.previous_state = starting_board
        self.most_recent_actn = None
        self.most_recent_state= None


    # follow an epsilon greedy policy to make a move mid-game
    # Returns STATE hash
    def move(self, state_hash):
        print('moving')       
        self.previous_state = state_hash
        
        if rnd.uniform() < self.eps:
            # random action
            act_hash = rand_move(state_hash)
            
        else: 
            # greedy action
            acts = self.Q[state_hash]
            act_hash = max(acts, key=acts.get)
        
        self.most_recent_actn = act_hash
        print('Most recent action (move):', self.most_recent_actn)
        
        state = de_hash(state_hash)
        act = de_hash(act_hash)
        new_state = [s - a for s,a in zip(state, act)]
        self.most_recent_state = get_hash(new_state)
        
        return get_hash(new_state)
        

            
        
    def bellman(self, reward):
         avail_acts = self.Q[self.previous_state]
         
         next_acts = self.Q[self.most_recent_state]
         
         max_next_vals = 0 # idk what to do about this value when you win?
         if 0 < len(next_acts): 
             max_next_vals = next_acts[max(next_acts, key=next_acts.get)]
         
         # this updates Q
         print('Most recent action (ball):', self.most_recent_actn)
         avail_acts[self.most_recent_actn] += self.alpha*(reward + self.gamma*max_next_vals + avail_acts[self.most_recent_actn])

        
    ## win_lose_flag: 1-won, 0-lose, None-midgame
    def learn(self, win_lose_flag, reward):
        # learning mid-game
        print('learn method')
        
        if 1 == win_lose_flag:
            self.games_played += 1
            self.games_won +=1 
            self.bellman(reward)
        elif 0 == win_lose_flag:
            self.games_played += 1
            self.bellman(reward)
        else:
            # mid-game, no reward
            self.bellman(0)
        


class PerfectAgent():
    def move(self, board): 
        # make an optimal move mid-game
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
        

