# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:04:05 2018

@author: Aditya
"""

MAX_COL = 6
MAX_ROW = 6
N_IMAGES = MAX_COL * MAX_ROW
IMAGE_SIZE = 2.15
GAP_LD = 0.085
GAP_HD = 0.85
LINEAR_REWARD = [-1, 0, 4, 7, 10, 13]
POWER_REWARD = [-1, 0, 2, 3, 6, 20]

LINEAR_REWARD_TEST = [-1, 0, 40, 70, 100, 130]
POWER_REWARD_TEST = [-1, 0, 20, 30, 60, 200]


MAX_ACTIONS = N_IMAGES + 1
CLICK_ACTION = N_IMAGES
MAX_STEPS = 15

#Properties
HIGH_DENSITY =  False
LOW_DENSITY =  True
POWER_FUNCTION =  False
LINEAR_FUNCTION = True

HIGH_LINEAR_TIME = 0.325
LOW_LINEAR_TIME = 0.310
HIGH_POWER_TIME = 0.350
LOW_POWER_TIME = 0.330