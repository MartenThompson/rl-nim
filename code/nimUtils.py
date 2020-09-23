# utility methods for agents
from functools import reduce
from operator import xor
import numpy.random as rnd


def board_bitsum(board):
    """
    Calculates bitsum (component-wise XOR sum) of board.

    Parameters
    ----------
    board : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    return(reduce(xor, board))


def rand_move(board):
    """
    Makes random legal move on board.
    Mututates baord by reference.
    """
    
    pile_size = 0
    while pile_size == 0: 
        pile = rnd.randint(0,len(board),1)[0]
        pile_size = board[pile]
    
    if pile_size == 1:
        board[pile] = 0
        return()
    
    take = rnd.randint(1,board[pile],1)[0]
    board[pile] -= take

    
    
    
    
    
    
    