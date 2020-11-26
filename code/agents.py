# Class definitions for agents

import csv
from datetime import datetime
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
    def __init__(self, name, starting_board_hash, init_Qval, eps, alpha, gamma):
        self.name = name
        self.games_played = 0
        self.games_won = 0
        
        # call Q(state) to get available actions. Call avail_acts(act_hash) to get value
        num_piles = len(de_hash(starting_board_hash))
        self.Q = initialize_Q(num_piles, init_Qval)
        
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.previous_state = starting_board_hash
        self.most_recent_actn = None
        self.most_recent_state= None
        self.optimal_moves = []
        self.optimal_per_game = []
        self.optimal_per_start = []
        self.winlose_per_start = []
        self.started = None


    # follow an epsilon greedy policy to make a move mid-game
    # Returns STATE hash (i.e. board)
    def move(self, state_hash):
        #print('moving')       
        self.previous_state = state_hash
        
        if rnd.uniform() < self.eps:
            # random action
            act_hash = rand_move(state_hash)
            
        else: 
            # greedy action
            acts = self.Q[state_hash]
            act_hash = max(acts, key=acts.get)
        
        self.most_recent_actn = act_hash
        #print('Most recent action (move):', self.most_recent_actn)
        
        state = de_hash(state_hash)
        act = de_hash(act_hash)
        new_state = [s - a for s,a in zip(state, act)]
        self.most_recent_state = get_hash(new_state)
        
        
        optimal = board_bitsum(new_state) == 0
        self.optimal_moves.append(optimal)
        
        return get_hash(new_state)
        

            
        
    def bellman(self, win_lose_flag, reward):
        
        if win_lose_flag == 0 or win_lose_flag == 1:
            # game over
            self.Q[self.previous_state][self.most_recent_actn] += self.alpha*(reward - self.Q[self.previous_state][self.most_recent_actn])
            return
        
        avail_acts = self.Q[self.previous_state]

        next_acts = self.Q[self.most_recent_state]
         
        max_next_vals = 0
        if 0 < len(next_acts): 
            max_next_vals = next_acts[max(next_acts, key=next_acts.get)]
         
        # this updates Q
        # print('Most recent action (ball):', self.most_recent_actn)
        avail_acts[self.most_recent_actn] += self.alpha*(reward + self.gamma*max_next_vals - avail_acts[self.most_recent_actn])
        #print(avail_acts[self.most_recent_actn])

        
    ## win_lose_flag: 1-won, 0-lose, None-midgame
    def learn(self, win_lose_flag, reward):
        # learning mid-game
        # print('learn method')
        
        if 1 == win_lose_flag:
            self.games_played += 1
            self.games_won +=1 
            self.bellman(win_lose_flag, reward)
            self.optimal_per_game.append(np.mean(self.optimal_moves))
            if self.started:
                self.optimal_per_start.append(np.mean(self.optimal_moves))
                self.winlose_per_start.append(1)
            
            self.optimal_moves = []
        elif 0 == win_lose_flag:
            self.games_played += 1
            self.bellman(win_lose_flag, reward)
            self.optimal_per_game.append(np.mean(self.optimal_moves))
            if self.started:
                self.optimal_per_start.append(np.mean(self.optimal_moves))
                self.winlose_per_start.append(0)
            
            self.optimal_moves = []
        else:
            # mid-game, no reward
            self.bellman(win_lose_flag, reward)
            
    def regularize(self, factor):
        all_vals = []
        
        for key, sub_dict in self.Q.items():
            for k,v in sub_dict.items():
                all_vals.append(v)
        
        max_val = abs(max(all_vals, key=abs))
        
        if max_val != 0:
            for key, sub_dict in self.Q.items():
                for k,v in sub_dict.items():
                    self.Q[key][k] = factor*(self.Q[key][k]/max_val)
        
        
            
    # Header: runtime(s), test accuracy, 0.0
    # Then: epoch, epoch loss, epoch accuracy
    def write(self):
        filename = 'out/QAgent_'+ self.name +  datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '.csv'
        file = open(filename, 'w+', newline ='')
        file_contents = np.zeros([len(self.optimal_per_start),1], dtype=float)
        file_contents[:,0] = self.optimal_per_start

        
        with file:
            write = csv.writer(file)
            write.writerows(file_contents)
            
    def report_perc(self):
        print('n')
        

class SARSAAgent():
    """
    on-policy
    """
    def __init__(self, name, num_piles, starting_board, init_Qval, eps, alpha, gamma):
        self.name = name
        self.games_played = 0
        self.games_won = 0
        # call Q(state) to get available actions. Call avail_acts(act_hash) to get value
        self.Q = initialize_Q(num_piles, init_Qval)
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.previous_state = starting_board
        self.most_recent_actn = None
        self.most_recent_state= None
        self.optimal_moves = []
        self.optimal_per_game = []
        self.optimal_per_start = []
        self.started = None


    # follow an epsilon greedy policy to make a move mid-game
    # Returns STATE hash (i.e. board)
    def move(self, state_hash):
        #print('moving')       
        self.previous_state = state_hash
        
        if rnd.uniform() < self.eps:
            # random action
            act_hash = rand_move(state_hash)
            
        else: 
            # greedy action
            acts = self.Q[state_hash]
            act_hash = max(acts, key=acts.get)
        
        self.most_recent_actn = act_hash
        #print('Most recent action (move):', self.most_recent_actn)
        
        state = de_hash(state_hash)
        act = de_hash(act_hash)
        new_state = [s - a for s,a in zip(state, act)]
        self.most_recent_state = get_hash(new_state)
        
        
        optimal = board_bitsum(new_state) == 0
        self.optimal_moves.append(optimal)
        
        return get_hash(new_state)
        

            
        
    def bellman(self, win_lose_flag, reward):
        
        if win_lose_flag == 0 or win_lose_flag == 1:
            # game over
            self.Q[self.previous_state][self.most_recent_actn] += self.alpha*(reward + self.Q[self.previous_state][self.most_recent_actn])
            return
        
        avail_acts = self.Q[self.previous_state]

        next_acts = self.Q[self.most_recent_state]
         
        max_next_vals = 0
        if 0 < len(next_acts): 
            max_next_vals = next_acts[max(next_acts, key=next_acts.get)]
         
        # this updates Q
        # print('Most recent action (ball):', self.most_recent_actn)
        avail_acts[self.most_recent_actn] += self.alpha*(reward + self.gamma*max_next_vals + avail_acts[self.most_recent_actn])
        #print(avail_acts[self.most_recent_actn])

        
    ## win_lose_flag: 1-won, 0-lose, None-midgame
    def learn(self, win_lose_flag, reward):
        # learning mid-game
        # print('learn method')
        
        if 1 == win_lose_flag:
            self.games_played += 1
            self.games_won +=1 
            self.bellman(win_lose_flag, reward)
            self.optimal_per_game.append(np.mean(self.optimal_moves))
            if self.started:
                self.optimal_per_start.append(np.mean(self.optimal_moves))
            
            self.optimal_moves = []
        elif 0 == win_lose_flag:
            self.games_played += 1
            self.bellman(win_lose_flag, reward)
            self.optimal_per_game.append(np.mean(self.optimal_moves))
            if self.started:
                self.optimal_per_start.append(np.mean(self.optimal_moves))
            
            self.optimal_moves = []
        else:
            # mid-game, no reward
            self.bellman(win_lose_flag, reward)
            
    def regularize(self, factor):
        all_vals = []
        
        for key, sub_dict in self.Q.items():
            for k,v in sub_dict.items():
                all_vals.append(v)
        
        max_val = abs(max(all_vals, key=abs))
        
        if max_val != 0:
            for key, sub_dict in self.Q.items():
                for k,v in sub_dict.items():
                    self.Q[key][k] = factor*(self.Q[key][k]/max_val)
        
        
            
    # Header: runtime(s), test accuracy, 0.0
    # Then: epoch, epoch loss, epoch accuracy
    def write(self):
        filename = 'out/QAgent_'+ self.name +  datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '.csv'
        file = open(filename, 'w+', newline ='')
        file_contents = np.zeros([len(self.optimal_per_start),1], dtype=float)
        file_contents[:,0] = self.optimal_per_start

        
        with file:
            write = csv.writer(file)
            write.writerows(file_contents)






class PerfectAgent():
    def __init__(self, name):
        self.name = name
        self.games_played = 0
        self.games_won = 0
        self.games_started = 0
        self.games_started_won = 0
        self.started = None
    
    # Returns STATE hash (i.e. board)
    def move(self, state_hash): 
        # make an optimal move mid-game
        # mutates board by reference
        
                
        # optimal moves are 
            # 1: taking the final pile to win
            # 2: reducing the board's bitsum to 0 when possible
        state = de_hash(state_hash)
        if sum(state) <= 0:
            sys.exit('Game already over or in illegal state')

        
        # 1
        if np.count_nonzero(state) == 1: 
            state[:] = np.zeros(len(state), dtype=int)      #.tolist()
            return(get_hash(state))
                
        # 2
        for i in range(len(state)):
            for j in (np.arange(state[i]) + 1):     # +1 because can't take 0
                temp_state = state[:]
                temp_state[i] -= j                  # take j items from pile i
                
                if board_bitsum(temp_state) == 0:
                    state[i] -= j
                    
                    return(get_hash(state))
                
        
        # no optimal move possible, move randomly
        act_hash = rand_move(state_hash)
        act = de_hash(act_hash)
        new_state = [s - a for s,a in zip(state, act)]
        return(get_hash(new_state))
    
    ## win_lose_flag: 1-won, 0-lose, None-midgame
    def learn(self, win_lose_flag, reward):
        if 1 == win_lose_flag:
            self.games_played += 1
            self.games_won +=1 
            if self.started:
                self.games_started += 1
                self.games_started_won +=1
                        
        elif 0 == win_lose_flag:            
            self.games_played += 1
            if self.started:
                self.games_started += 1

        #else: # mid-game, no reward

    
    def write(self):
        filename = 'out/PerfectAgent_'+ self.name +  datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '.csv'
        file = open(filename, 'w+', newline ='')
        file_contents = np.zeros([2,1], dtype=int)
        file_contents[0,0] = self.games_started
        file_contents[1,0] = self.games_started_won

        
        with file:
            write = csv.writer(file)
            write.writerows(file_contents)

