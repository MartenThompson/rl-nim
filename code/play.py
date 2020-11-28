# Primary script in project

from agents import QAgent
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


def QvQgrid():
    """
    Simulate 10 repetitions of 75000 training games between two Q-learners 
    across a grid of hyperparameters. Write out perc. optimal moves and
    win/lose rates for each game an agent was player 1.
    """
    rnd.seed(513)
    
    N_GAMES = 75000 
    REPS = 5
    Q_init = 0.0
    
    epsilons = [0.1,0.3,0.5,0.7,0.9]
    epsilons = [0.7,0.9]
    alphas = [0.1,0.3,0.5,0.7,0.9]
    # alphas = [0.1]
    gammas = [-0.1,-0.3,-0.5,-0.7,-0.9]
    # gammas = [-0.1]
    
    setting = [[e,a,g] for e in epsilons for a in alphas for g in gammas] 
    
    for s in range(len(setting)):
        params = setting[s]
        epsilon = params[0]
        alpha = params[1]
        gamma = params[2]
        
        print('e:', epsilon, ' a:', alpha, ' g:', gamma)
        
        p1_opt_percs = []
        p1_winlose = []
        p2_opt_percs = []
        p2_winlose = []
        
        for i in range(REPS):
            bd = rnd.randint(0,9,3).tolist()
            if sum(bd) >= 0:
                starting_board_hash = get_hash(rnd.randint(0,9,3).tolist())
            else:
                starting_board_hash = get_hash([4,4,4]) # one in million chance this will be needed
            
            p1 = QAgent('p1', starting_board_hash, Q_init, epsilon, alpha, gamma)
            p2 = QAgent('p2', starting_board_hash, Q_init, epsilon, alpha, gamma)
            
            [p1_stats, p2_stats] = train_agents(N_GAMES, p1, p2, starting_board_hash, 1, -1, False)
            p1_opt_percs.append(p1_stats[0])
            p1_winlose.append(p1_stats[1])
            p2_opt_percs.append(p2_stats[0])
            p2_winlose.append(p2_stats[1])            
        
        filename = 'final/' + 'optimal_moves' + str(epsilon) + str(alpha) + str(gamma) +'vSelfAll_'+ datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '.csv'
        file = open(filename, 'w+', newline ='')
        file_contents = p1_opt_percs + p2_opt_percs
    
        with file:
            write = csv.writer(file)
            write.writerows(file_contents)
            
        filename = 'final/' + 'wins' + str(epsilon) + str(alpha) + str(gamma) +'vSelfAll_'+ datetime.now().strftime("%Y_%m_%d-%H_%M_%S") + '.csv'
        file = open(filename, 'w+', newline ='')
        file_contents = p1_winlose + p2_winlose
        
        with file:
            write = csv.writer(file)
            write.writerows(file_contents)
    
    print('learning complete')


def vis_learning():
    """
    Write out Q table for agent as it learns.
    """
    rnd.seed(444)
    
    N_GAMES = 1000
    
    starting_board_hash = get_hash([2,2])
    p1 = QAgent('p1', starting_board_hash, 0.0, 0.1, 0.3, -0.5)
    p2 = QAgent('p2', starting_board_hash, 0.0, 0.1, 0.3, -0.5)
    
    # values in Q table over time
    Qseries = pd.DataFrame(columns=['01-00', 
                                    '02-00', '02-01',
                                    '10-00',
                                    '11-01', '11-10',
                                    '12-02', '12-10', '12-11',
                                    '20-00', '20-10',
                                    '21-01', '21-11', '21-20',
                                    '22-02', '22-12', '22-20', '22-21'])
    
    
    for i in range(N_GAMES):
        play_nim(p1, p2, starting_board_hash, 1, -1)
        temp = []
        temp.append(p1.Q['0, 1']['0, 1'])
        temp.append(p1.Q['0, 2']['0, 2'])
        temp.append(p1.Q['0, 2']['0, 1'])
        temp.append(p1.Q['1, 0']['1, 0'])    
        temp.append(p1.Q['1, 1']['1, 0'])
        temp.append(p1.Q['1, 1']['0, 1'])        
        temp.append(p1.Q['1, 2']['1, 0'])
        temp.append(p1.Q['1, 2']['0, 2'])
        temp.append(p1.Q['1, 2']['0, 1'])
        temp.append(p1.Q['2, 0']['2, 0'])
        temp.append(p1.Q['2, 0']['1, 0'])
        temp.append(p1.Q['2, 1']['2, 0'])
        temp.append(p1.Q['2, 1']['1, 0'])
        temp.append(p1.Q['2, 1']['0, 1'])
        temp.append(p1.Q['2, 2']['2, 0'])
        temp.append(p1.Q['2, 2']['1, 0'])
        temp.append(p1.Q['2, 2']['0, 2'])
        temp.append(p1.Q['2, 2']['0, 1'])
        
        Qseries.loc[i] = temp
    
    
    Qseries.to_csv('final/Qvis/Qseries.csv', index=False)
    print('complete')
    
    
    
    


if __name__ == "__main__":
    # QvQgrid()
    # BestvRand()
    vis_learning()
