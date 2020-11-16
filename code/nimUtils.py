# utility methods for agents
from functools import reduce
from operator import xor
import numpy as np
import numpy.random as rnd
from statesActions import get_hash
from statesActions import de_hash

def board_bitsum(board):
    """
    Calculates bitsum (component-wise XOR sum) of board.
    
    An optimal move results in a bitsum == 0

    Parameters
    ----------
    board : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return(reduce(xor, board))


def rand_moveold(state_hash):
    """
    Finds a random legal move on board.
    Returns state of new board
    """
    state = de_hash(state_hash)
    new_state = []
    new_state[:] = state
    
    pile_size = 0
    while pile_size == 0: 
        pile = rnd.randint(0,len(state),1)[0]               # TODO: not v efficient
        pile_size = state[pile]
    
    if pile_size == 1:
        new_state[pile] = 0
        return get_hash(new_state)
    
    take = rnd.randint(1,state[pile],1)[0]
    new_state[pile] = state[pile] - take
    return get_hash(new_state)
        
    
    

def rand_move(board_state_hash):
    state = de_hash(board_state_hash)
    
    act = np.zeros(len(state), dtype=int).tolist()
    
    pile_size = 0
    pile = 0
    while 0 == pile_size:
        pile = rnd.randint(0,len(state),1)[0]
        pile_size = state[pile]
        
    if 1 == pile_size: 
        act[pile] = 1
        return(get_hash(act))
    
    take = rnd.randint(1,state[pile],1)[0]
    act[pile] = take
    return(get_hash(act))