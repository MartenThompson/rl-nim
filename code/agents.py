# Class definitions for agents

import csv
from datetime import datetime
import numpy.random as rnd
import numpy as np
import sys

from nimUtils import board_bitsum
from nimUtils import rand_move
from statesActions import initialize_Q
from statesActions import initialize_V
from statesActions import get_hash
from statesActions import de_hash

from scipy.stats import gamma
from scipy.stats import norm 

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
        

class QtAgent():
    """
    off-policy, so moving is independent of learning
    """
    def __init__(self, name, starting_board_hash, init_Qval, eps, alpha, gamma, eta):
        self.name = name
        self.games_played = 0
        self.games_won = 0
        
        # call Q(state) to get available actions. Call avail_acts(act_hash) to get value
        num_piles = len(de_hash(starting_board_hash))
        self.Q = initialize_Q(num_piles, init_Qval)
        
        self.eps = eps
        self.eps_0 = eps
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta
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
        avail_acts[self.most_recent_actn] += self.alpha*(reward + self.gamma*max_next_vals - avail_acts[self.most_recent_actn])
        

    def update_eps(self):
        self.eps = self.eps_0*np.exp(-self.eta*self.games_played)
        
        
    def learn(self, win_lose_flag, reward):
    # win_lose_flag: 1-won, 0-lose, None-midgame    
        if 1 == win_lose_flag:
            # game over, won
            self.games_played += 1
            self.update_eps()
            self.games_won +=1 
            self.bellman(win_lose_flag, reward)
            self.optimal_per_game.append(np.mean(self.optimal_moves))
            if self.started:
                self.optimal_per_start.append(np.mean(self.optimal_moves))
                self.winlose_per_start.append(1)
            
            self.optimal_moves = []
        elif 0 == win_lose_flag:
            # game over, lost
            self.games_played += 1
            self.update_eps()
            self.bellman(win_lose_flag, reward)
            self.optimal_per_game.append(np.mean(self.optimal_moves))
            if self.started:
                self.optimal_per_start.append(np.mean(self.optimal_moves))
                self.winlose_per_start.append(0)
            
            self.optimal_moves = []
        else:
            # mid-game, no reward
            self.bellman(win_lose_flag, reward)
             
            
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

class BayesAgent():
    """
    on-policy
    """
    def __init__(self, name, starting_board_hash, mu_0, lamb_0, alpha_0, beta_0, discount):
        self.name = name
        self.games_played = 0
        self.games_won = 0
        
        # call Q(state) to get available actions. Call avail_acts(act_hash) to get value
        num_piles = len(de_hash(starting_board_hash))
        self.V = initialize_V(num_piles, mu_0, lamb_0, alpha_0, beta_0)
        
        self.previous_state = None
        #self.mu = mu_0
        #self.lamb = lamb_0
        #self.alpha = alpha_0
        #self.beta = beta_0
        self.discount = discount
        
        self.sa_history = [] # moves (state, action) from current game
        
        #self.previous_state = starting_board_hash
        #self.most_recent_actn = None
        #self.most_recent_state= None
        self.optimal_moves = [] # individual game
        self.optimal_per_game = [] # running log
        self.optimal_per_start = []
        self.winlose_per_start = []
        self.started = None


    # follow an epsilon greedy policy to make a move mid-game
    # Returns STATE hash (i.e. board)
    def move(self, state_hash):
        
        avail_acts = self.V[state_hash]
        v_all = np.zeros(len(avail_acts))
        a_all = np.empty(len(avail_acts), dtype='object') # hold strings (hashes)
        v = 0
        
        for a, h in avail_acts.items():
            m = norm.rvs(loc=h[0], scale=1/h[1], size=1)[0]
            t = gamma.rvs(a=h[2], scale=h[3], size=1)[0]
            v_all[v] = norm.rvs(loc=m, scale=1/t, size=1)[0]
            a_all[v] = a
            v += 1
        
        act_hash = a_all[max(range(len(v_all)), key=v_all.__getitem__)]
        
        state = de_hash(state_hash)
        act = de_hash(act_hash)
        new_state = [s - a for s,a in zip(state, act)]
        
        
        optimal = board_bitsum(new_state) == 0
        self.optimal_moves.append(optimal)
        
        self.sa_history.append([state_hash, act_hash])
        
        return get_hash(new_state)
        

            
        
    def update_V(self, win_lose_flag, reward):
        T = len(self.sa_history)
        
        for turn in range(T):
            r = reward * (self.discount**(T-turn))
            state = self.sa_history[turn][0]
            act = self.sa_history[turn][1]
            [mu_t, lamb_t, alpha_t, beta_t] = self.V[state][act]
            
            self.V[state][act] = [(mu_t*lamb_t + r)/(lamb_t + 1),
                             lamb_t + 1,
                             alpha_t + 0.5,
                             beta_t + 0.5 + (lamb_t)/(lamb_t + 1)*((r - mu_t)**2)/2]
    

        
        
    def learn(self, win_lose_flag, reward):
    # win_lose_flag: 1-won, 0-lose, None-midgame    
        if 1 == win_lose_flag:
            # game over, won
            self.games_played += 1
            self.games_won +=1 
            self.update_V(win_lose_flag, reward)
            self.optimal_per_game.append(np.mean(self.optimal_moves))
            
            if self.started:
                self.optimal_per_start.append(np.mean(self.optimal_moves))
                self.winlose_per_start.append(1)
            
            self.optimal_moves = []
            self.sa_history = []
        elif 0 == win_lose_flag:
            # game over, lost
            self.games_played += 1
            self.update_V(win_lose_flag, reward)
            self.optimal_per_game.append(np.mean(self.optimal_moves))
            
            if self.started:
                self.optimal_per_start.append(np.mean(self.optimal_moves))
                self.winlose_per_start.append(0)
            
            self.optimal_moves = []
            self.sa_history = []
        #else:
            # mid-game, no learning
             
            
    # Header: runtime(s), test accuracy, 0.0
    # Then: epoch, epoch loss, epoch accuracy
    def write(self):
        filename = 'out/BayesAgent_'+ self.name +  datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '.csv'
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
    
    # Returns state hash (i.e. board)
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





