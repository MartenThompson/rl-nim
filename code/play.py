# Methods to conduct a single game of nim

from agents import QAgent
from statesActions import get_hash
from statesActions import de_hash

import sys
import random as rnd
import numpy as np

# board should be list of ints 0-9
def play_nim(p1, p2, board, win_reward, lose_reward):
    """
    Simulate a single complete game between two agents. 
    Mutate them appropriately during and after it ends.
    """
    
    if sum(de_hash(board)) <= 0:
        sys.exit('Illegal starting board')
    
    print('Starting board:', board)
    
    turns = 0
    while True: 
        if 0 == turns:
            new_board_hash = p1.move(board)
        else:
            new_board_hash = p1.move(new_board_hash)
            
        print('P1:', new_board_hash)
        
        if 0 == sum(de_hash(new_board_hash)):
            # P1 wins
            print('P1 won!')
            p1.learn(1, win_reward)
            
            if 0 != turns:
                # can't teach it if it never got to move :(
                p2.learn(0, lose_reward) 
            
            break
        
        new_board_hash = p2.move(new_board_hash)
        print('P2:', new_board_hash)
        
        if 0 == sum(de_hash(new_board_hash)):
            # P2 wins
            print('P2 won!')
            p1.learn(0, lose_reward)
            p2.learn(1, win_reward)
            break
        
        # no one has one yet
        p1.learn(None, 0)
        p2.learn(None, 0)
        turns += 1
        print('round complete')
        
    print('game over')



def main():
    num_piles = 3
    Q_init = 0.0
    
    starting_board = get_hash([2,2,2])
    p1 = QAgent(num_piles, starting_board, Q_init, 0.5)
    p2 = QAgent(num_piles, starting_board, Q_init, 0.1)

    
    play_nim(p1, p2, starting_board, 1, -1)

if __name__ == "__main__":
    main()
