# Methods to conduct a single game of nim

from agents import QAgent
from agents import PerfectAgent
from statesActions import get_hash
from statesActions import de_hash

import sys
import numpy.random as rnd
import numpy as np

# board should be list of ints 0-9
def play_nim(p1, p2, board_hash, win_reward, lose_reward):
    """
    Simulate a single complete game between two agents. 
    Mutate them appropriately during and after it ends.
    """
    
    if sum(de_hash(board_hash)) <= 0:
        sys.exit('Illegal starting board')
    
    # print('Starting board:', board)
    
    turns = 0
    while True: 
        if 0 == turns:
            new_board_hash = p1.move(board_hash)
        else:
            new_board_hash = p1.move(new_board_hash)
            
        #print('P1:', new_board_hash)
        
        if 0 == sum(de_hash(new_board_hash)):
            # P1 wins
            # print('P1 won!')
            p1.learn(1, win_reward)
            
            if 0 != turns:
                # can't teach it if it never got to move :(
                p2.learn(0, lose_reward) 
            
            break
        
        new_board_hash = p2.move(new_board_hash)
        #print('P2:', new_board_hash)
        
        if 0 == sum(de_hash(new_board_hash)):
            # P2 wins
            # print('P2 won!')
            p1.learn(0, lose_reward)
            p2.learn(1, win_reward)
            break
        
        # no one has one yet
        p1.learn(None, 0)
        p2.learn(None, 0)

        turns += 1
        #print('round complete')
        
    #print('game over')



def main():
    n_games = 60000 # int(5e3)
    num_piles = 3
    Q_init = 0.0
    
    rnd.seed(5)
    
    # starting_board = get_hash(rnd.randint(0,9,3).tolist())
    starting_board = get_hash([4,5,8])
    
    #p1 = QAgent('p102', num_piles, starting_board, Q_init, 0.1) 
    p1 = QAgent('p1', num_piles, starting_board, Q_init, 0.1, 0.1, -0.5)
    # p2 = PerfectAgent('perfect')
    p2 = QAgent('p2', num_piles, starting_board, Q_init, 0.2, 0.1, 0.9)
    
    """
    for i in range(n_games):
        p1.started = True
        p2.started = False
        
        if i > 900:
            print('debug')
        play_nim(p1, p2, starting_board, 1, -1)

    """
    for i in range(n_games):
        if i %1000 == 0:
            #p1.regularize(10)
            #p2.regularize(10)
            print(i)
        
        if rnd.uniform(0,1,1)[0] > 0.5: 
            p1.started = True
            p2.started = False
            play_nim(p1, p2, starting_board, 1, -1)    
        else:
            p1.started = False
            p2.started = True
            play_nim(p2, p1, starting_board, 1, -1)
    
    p1.write()
    p2.write()
    
    print('training complete')

if __name__ == "__main__":
    main()
