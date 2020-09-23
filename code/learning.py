# Learning (training) script: simulate many matches

from agents import QAgent
import random as rnd
import numpy as np


n = 1                                   # number of training matches

who_starts = rnd.randrange(2)
print(who_starts)

if who_starts == 0:                     # Q agent starts
    play_nim(agent_0, agent_1)

else: 
    play_nim(agent_1, agent_0)
        






