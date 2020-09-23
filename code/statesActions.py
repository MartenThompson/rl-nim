"""
Created on Wed Sep 23 12:57:20 2020

@author: MartenThompson
"""

import numpy as np

def get_hash(sa):
    # h = 0
    # for i in range(len(sa)):
    #     h += (10**i)*(sa[i] + 1)
        
    # return(h)
    string = str(sa)                            # '[1,2,5]'
    return(string[1:-1])                        # '1,2,5'

def de_hash(hash_string):
    sa = hash_string.split(',')


def initialize_available_actions(num_piles):
    MAX_PILE_SIZE = 9
    
    # First, calculate all available states
    all_states = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]
    
    if num_piles > 1:
        for i in range(num_piles-1):
            all_states = recur2(all_states)
            
    # Second, make dictionary {state: available actions}
    all_avail_actions = {}
    for i in range(len(all_states)):
        all_avail_actions[get_hash(all_states[i])] = avail_actions(all_states[i])
    
    return(all_avail_actions)
   

def avail_actions(state):
    ret = []
    for i in range(len(state)):
        if state[i] > 0:
            for j in (1+np.arange(state[i])):
                temp = []
                temp[:] = state
                temp[i] = state[i] - j
                ret.append(temp)
    
    return(ret)


def recur2(l):
    ret = []
    for i in range(len(l)):
        for j in range(10):
            temp = []
            temp[:] = l[i]
            temp.append(j)
            ret.append(temp)
            
    return(ret)