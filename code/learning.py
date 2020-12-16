# Primary script in project. 
# Orchestrates the simulations used in report.

import numpy as np
import numpy.random as rnd
import pandas as pd
from agents import QAgent
from agents import QtAgent
from agents import BayesAgent
from statesActions import get_hash

import copy
import csv
from datetime import datetime

from play import play_nim
from play import train_agents

def log_contents(file_name, file_contents):
    file_name = file_name + datetime.now().strftime('%Y_%m_%d') + '.csv'
    file = open(file_name, 'w+', newline ='')

    with file:
        write = csv.writer(file)
        write.writerows(file_contents)


def log_Q_learn(n_games, p1, p2, win_reward, lose_reward):
    """
    Write out Q table for p1 as it learns on board [2,2]. P1 always moves first.
    """
    
    starting_board_hash = get_hash([2,2])
    
    # values in Q table over time
    Qseries = pd.DataFrame(columns=['01-00', 
                                    '02-00', '02-01',
                                    '10-00',
                                    '11-01', '11-10',
                                    '12-02', '12-10', '12-11',
                                    '20-00', '20-10',
                                    '21-01', '21-11', '21-20',
                                    '22-02', '22-12', '22-20', '22-21'])
    
    
    for i in range(n_games):
        play_nim(p1, p2, starting_board_hash, win_reward, lose_reward)
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
    
    
    Qseries.to_csv('../final/Q_vis/' + p1.name + '_series.csv', index=False)


def vis_learning():
    rnd.seed(808)
    
    N_GAMES = 50000
    
    # conservative players
    starting_board_hash = get_hash([2,2])
    p1 = QAgent('conservative', starting_board_hash, 0.0, 0.1, 0.1, -0.1)
    p2 = QAgent('conservative', starting_board_hash, 0.0, 0.1, 0.1, -0.1)
    
    log_Q_learn(N_GAMES, p1, p2, 1, -1)
    
    # aggressive players
    p1 = QAgent('aggressive', starting_board_hash, 0.0, 0.1, 0.9, -0.9)
    p2 = QAgent('aggressive', starting_board_hash, 0.0, 0.1, 0.9, -0.9)
    log_Q_learn(N_GAMES, p1, p2, 1, -1)
    
    print('complete')


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
    
    epsilons = [0.1, 0.3, 0.5]
    alphas = [0.1,0.3,0.5,0.7,0.9]
    gammas = [-0.1,-0.3,-0.5,-0.7,-0.9]
    
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
        
        file_name = '../final/QvQ/QvQ_optimal_moves' + str(epsilon) + str(alpha) + str(gamma) +'vSelfAll_'
        file_contents = p1_opt_percs + p2_opt_percs
        log_contents(file_name, file_contents)
            
        file_name = '../final/QvQ/QvQ_wins' + str(epsilon) + str(alpha) + str(gamma) +'vSelfAll_'
        file_contents = p1_winlose + p2_winlose
        log_contents(file_name, file_contents)

    print('learning complete')
    
    
def QtvQtgrid():
    """
    Simulate 10 repetitions of 75000 training games between two Qt-learners 
    across a grid of hyperparameters. Write out perc. optimal moves and
    win/lose rates for each game an agent was player 1.
    """
    rnd.seed(2001)
    
    N_GAMES = 75000 
    REPS = 5
    Q_init = 0.0
    
    epsilons = [0.1, 0.5, 0.9]
    alphas = [0.1, 0.3]
    gammas = [-0.5, -0.9]
    etas = [0.001, 0.0001, 0.00001]
    
    setting = [[e,a,g,n] for e in epsilons for a in alphas for g in gammas for n in etas] 
    
    for s in range(len(setting)):
        params = setting[s]
        epsilon = params[0]
        alpha = params[1]
        gamma = params[2]
        eta = params[3]
        
        print('e:', epsilon, ' a:', alpha, ' g:', gamma, ' eta:', eta)
        
        p1_opt_percs = []
        p1_winlose = []
        p2_opt_percs = []
        p2_winlose = []
        
        for i in range(REPS):
            bd = rnd.randint(0,9,3).tolist()
            if sum(bd) >= 0:
                starting_board_hash = get_hash(rnd.randint(0,9,3).tolist())
            else:
                starting_board_hash = get_hash([5,5,5]) # one in million chance this will be needed
            
            p1 = QtAgent('p1', starting_board_hash, Q_init, epsilon, alpha, gamma, eta)
            p2 = QtAgent('p2', starting_board_hash, Q_init, epsilon, alpha, gamma, eta)
            
            [p1_stats, p2_stats] = train_agents(N_GAMES, p1, p2, starting_board_hash, 1, -1, False)
            p1_opt_percs.append(p1_stats[0])
            p1_winlose.append(p1_stats[1])
            p2_opt_percs.append(p2_stats[0])
            p2_winlose.append(p2_stats[1])            
        
        file_name = '../final/QtvQt/QtvQt_optimal_moves' + str(epsilon) + str(alpha) + str(gamma) + str(eta) +'vSelfAll_'
        file_contents = p1_opt_percs + p2_opt_percs
        log_contents(file_name, file_contents)
            
        file_name = '../final/QtvQt/QtvQt_wins' + str(epsilon) + str(alpha) + str(gamma) + str(eta) +'vSelfAll_'
        file_contents = p1_winlose + p2_winlose
        log_contents(file_name, file_contents)
    
    print('learning complete')
    
    
def bayesAgents():
    n_games = 25000
    reps = 5
    mu_0 = 0
    lamb_0 = 2
    alpha_0 = 1/2
    beta_0 = 1
    #discounts = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    discounts = [0.1, 0.5, 0.9, 0.99]
    
    for discount in discounts:
        print('d:', discount)
        p1_opt_percs = []
        p1_winlose = []
        p2_opt_percs = []
        p2_winlose = []
        for i in range(reps):
            print(i+1, '/', reps)
            
            bd = rnd.randint(0,9,3).tolist()
            if sum(bd) >= 0:
                starting_board_hash = get_hash(rnd.randint(0,9,3).tolist())
            else:
                starting_board_hash = get_hash([5,5,5]) # one in million chance this will be needed
            
            p1 = BayesAgent('p1',  starting_board_hash, mu_0, lamb_0, alpha_0, beta_0, discount)
            p2 = BayesAgent('p2',  starting_board_hash, mu_0, lamb_0, alpha_0, beta_0, discount)
            
            [p1_stats, p2_stats] = train_agents(n_games, p1, p2, starting_board_hash, 1, -1, False)
            p1_opt_percs.append(p1_stats[0])
            p1_winlose.append(p1_stats[1])
            p2_opt_percs.append(p2_stats[0])
            p2_winlose.append(p2_stats[1])            
        
        file_name = '../final/Bayes/Bayes_optimal_moves' + str(discount) + 'vSelfAll_'
        file_contents = p1_opt_percs + p2_opt_percs
        log_contents(file_name, file_contents)
            
        file_name = '../final/Bayes/Bayes_wins' + str(discount) +'vSelfAll_'
        file_contents = p1_winlose + p2_winlose
        log_contents(file_name, file_contents)

    print('learning complete')
    

def log_Bayes_learn(n_games, p1, p2, win_reward, lose_reward):
    """
    Write out Q table for p1 as it learns on board [2,2]. P1 always moves first.
    """
    
    starting_board_hash = get_hash([2,2])
    
    # values of mu over time
    # to match writeup, this should be 'm_series'
    mu_series = pd.DataFrame(columns=['01-00', 
                                    '02-00', '02-01',
                                    '10-00',
                                    '11-01', '11-10',
                                    '12-02', '12-10', '12-11',
                                    '20-00', '20-10',
                                    '21-01', '21-11', '21-20',
                                    '22-02', '22-12', '22-20', '22-21'])
    # to match writeup, this should be 'nu_series'
    tau_series = pd.DataFrame(columns=['01-00', 
                                    '02-00', '02-01',
                                    '10-00',
                                    '11-01', '11-10',
                                    '12-02', '12-10', '12-11',
                                    '20-00', '20-10',
                                    '21-01', '21-11', '21-20',
                                    '22-02', '22-12', '22-20', '22-21'])
    alpha_series = pd.DataFrame(columns=['01-00', 
                                    '02-00', '02-01',
                                    '10-00',
                                    '11-01', '11-10',
                                    '12-02', '12-10', '12-11',
                                    '20-00', '20-10',
                                    '21-01', '21-11', '21-20',
                                    '22-02', '22-12', '22-20', '22-21'])
    beta_series = pd.DataFrame(columns=['01-00', 
                                    '02-00', '02-01',
                                    '10-00',
                                    '11-01', '11-10',
                                    '12-02', '12-10', '12-11',
                                    '20-00', '20-10',
                                    '21-01', '21-11', '21-20',
                                    '22-02', '22-12', '22-20', '22-21'])
    
    
    for i in range(n_games):
        play_nim(p1, p2, starting_board_hash, win_reward, lose_reward)
        temp = []
        temp.append(p1.V['0, 1']['0, 1'])
        temp.append(p1.V['0, 2']['0, 2'])
        temp.append(p1.V['0, 2']['0, 1'])
        temp.append(p1.V['1, 0']['1, 0'])    
        temp.append(p1.V['1, 1']['1, 0'])
        temp.append(p1.V['1, 1']['0, 1'])        
        temp.append(p1.V['1, 2']['1, 0'])
        temp.append(p1.V['1, 2']['0, 2'])
        temp.append(p1.V['1, 2']['0, 1'])
        temp.append(p1.V['2, 0']['2, 0'])
        temp.append(p1.V['2, 0']['1, 0'])
        temp.append(p1.V['2, 1']['2, 0'])
        temp.append(p1.V['2, 1']['1, 0'])
        temp.append(p1.V['2, 1']['0, 1'])
        temp.append(p1.V['2, 2']['2, 0'])
        temp.append(p1.V['2, 2']['1, 0'])
        temp.append(p1.V['2, 2']['0, 2'])
        temp.append(p1.V['2, 2']['0, 1'])
        
        temp = np.array(temp)
        mu_series.loc[i] = temp[:,0]
        tau_series.loc[i] = temp[:,1]
        alpha_series.loc[i] = temp[:,2]
        beta_series.loc[i] = temp[:,3]
    
    mu_series.to_csv('../final/Bayes_vis/' + p1.name + '_mu_series.csv', index=False)
    tau_series.to_csv('../final/Bayes_vis/' + p1.name + '_tau_series.csv', index=False)
    alpha_series.to_csv('../final/Bayes_vis/' + p1.name + '_alpha_series.csv', index=False)
    beta_series.to_csv('../final/Bayes_vis/' + p1.name + '_beta_series.csv', index=False)

    
def bayesVis():
    rnd.seed(898)
    mu_0 = 0
    lamb_0 = 2
    alpha_0 = 1/2
    beta_0 = 1
    N_GAMES = 10000
    
    # conservative learner
    discount = 0.5
    starting_board_hash = get_hash([2,2])
    p1 = BayesAgent('conservative',  starting_board_hash, mu_0, lamb_0, alpha_0, beta_0, discount)
    p2 = BayesAgent('conservative',  starting_board_hash, mu_0, lamb_0, alpha_0, beta_0, discount)
    
    log_Bayes_learn(N_GAMES, p1, p2, 1, -1)
    
    # aggressive learner
    discount = 0.9
    p1 = BayesAgent('aggressive',  starting_board_hash, mu_0, lamb_0, alpha_0, beta_0, discount)
    p2 = BayesAgent('aggressive',  starting_board_hash, mu_0, lamb_0, alpha_0, beta_0, discount)
    log_Bayes_learn(N_GAMES, p1, p2, 1, -1)
    
    print('complete')


def BestvEachOther():
    n_games = 25000
    reps = 5
    mu_0 = 0
    lamb_0 = 2
    alpha_0 = 1/2
    beta_0 = 1

    dummy_board = get_hash([1,2,3]) # will be replaced each game below
    players = [QAgent('best_Q', dummy_board, 0.0, 0.1, 0.1, -0.9), 
               QtAgent('best_Qt', dummy_board, 0.0, 0.9, 0.1, -0.9, 0.0001),
               BayesAgent('best_Bayes', dummy_board, mu_0, lamb_0, alpha_0, beta_0, 0.9),
               QAgent('best_random', dummy_board, 0.0, 1, 0, 0)]
    
    versus = [[p1,copy.deepcopy(p2)] for p1 in players for p2 in players]
    
    for v in versus:
        p1 = v[0]
        p2 = v[1]
        print(p1.name, ' vs ', p2.name)
              
        p1_opt_percs = []
        p1_winlose = []
        p2_opt_percs = []
        p2_winlose = []
        
        for i in range(reps):
            print(i+1, '/', reps)
            
            bd = rnd.randint(0,9,3).tolist()
            if sum(bd) >= 0:
                starting_board_hash = get_hash(rnd.randint(0,9,3).tolist())
            else:
                starting_board_hash = get_hash([5,5,5]) 
            
            
            p1.previous_state = starting_board_hash
            p2.previous_state = starting_board_hash
            
            [p1_stats, p2_stats] = train_agents(n_games, p1, p2, starting_board_hash, 1, -1, False)
            p1_opt_percs.append(p1_stats[0])
            p1_winlose.append(p1_stats[1])
            p2_opt_percs.append(p2_stats[0])
            p2_winlose.append(p2_stats[1])            
        
        file_name = '../final/PvP/opt_moves' + p1.name + 'VS' + p2.name
        file_contents = p1_opt_percs
        log_contents(file_name, file_contents)
        
        file_name = '../final/PvP/opt_moves' + p2.name + 'VS' + p1.name
        file_contents = p2_opt_percs
        log_contents(file_name, file_contents)
            
        file_name = '../final/PvP/wins' + p1.name + 'VS' + p2.name
        file_contents = p1_winlose
        log_contents(file_name, file_contents)
        
        file_name = '../final/PvP/wins' + p2.name + 'VS' + p1.name
        file_contents = p2_winlose
        log_contents(file_name, file_contents)



    print('learning complete')        

if __name__ == "__main__":
    vis_learning()      # 15m
    # QvQgrid()           # 5h
    # QtvQtgrid()         # 20m
    # bayesAgents()       # 2.5h
    # bayesVis()          # 20m
    # BestvEachOther()    # 1.5h 
