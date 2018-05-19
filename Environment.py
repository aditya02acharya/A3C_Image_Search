# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:10:53 2018

@author: Aditya
"""
from DisplayGenerator import DisplayGenerator
from ObservationModel import ObservationModel
from UtilityFunctions import UtilityFunctions
from GlobalConstants import *
import numpy as np
import random

class Environment(object):
    
    def __init__(self):	
        self.steps = 0
        self.generator = DisplayGenerator()
        self.model = ObservationModel()
        self.utility = UtilityFunctions()
        self.current_display = None
        self.focus = None
        self.observation = None
        self.total_time = 0.00001
        self.prev_action = -1
        self.selected_feature = 0
    
    def step(self, action):
        self.steps += 1
        done = False
        
        if action < N_IMAGES:
            self.observation = self.model.sample(self.current_display, action)
            self.focus = np.eye(N_IMAGES)[action]
                
            self.prev_action = action
        
        if action < N_IMAGES:
            if LINEAR_FUNCTION and HIGH_DENSITY:
                duration = HIGH_LINEAR_TIME
            elif LINEAR_FUNCTION and LOW_DENSITY:
                duration = LOW_LINEAR_TIME
            elif POWER_FUNCTION and HIGH_DENSITY:
                duration = HIGH_POWER_TIME
            elif POWER_FUNCTION and LOW_DENSITY:
                duration = LOW_POWER_TIME
                
            if not (self.prev_action == -1):
                e = self.utility.get_eccentricity((self.prev_action/MAX_COL),(self.prev_action%MAX_COL),(action/MAX_COL),(action%MAX_COL))
                reward = (duration + 0.027 * e) * -1
            else:
                reward = duration * -1             

        elif action == CLICK_ACTION:
            if self.prev_action == -1:
                reward = POWER_REWARD[5] * -1
            else:
                features = self.current_display[int(self.prev_action/MAX_COL)][int(self.prev_action%MAX_COL)].n_features()
                if POWER_FUNCTION:
                    reward = POWER_REWARD[features]
                else:
                    reward = LINEAR_REWARD[features]
                
        
        
        if action == CLICK_ACTION:
            done = True
        
        if self.steps >= MAX_STEPS:
            if action < N_IMAGES:
                features = self.current_display[int(action/MAX_COL)][int(action%MAX_COL)].n_features()
            else:
                features = self.current_display[int(self.prev_action/MAX_COL)][int(self.prev_action%MAX_COL)].n_features()
                
            if POWER_FUNCTION:
                reward = POWER_REWARD[features]
            else:
                reward = LINEAR_REWARD[features]
            done = True
            
        
        return self.observation.flatten(), self.focus.flatten(), reward, done
    
    def test_step(self, action):
        self.steps += 1
        done = False
        
        if action < N_IMAGES:
            self.observation = self.model.sample(self.current_display, action)
            self.focus = np.eye(N_IMAGES)[action]
                
            self.prev_action = action
        
        if action < N_IMAGES:
            if LINEAR_FUNCTION and HIGH_DENSITY:
                duration = HIGH_LINEAR_TIME*1000.0
            elif LINEAR_FUNCTION and LOW_DENSITY:
                duration = LOW_LINEAR_TIME*1000.0
            elif POWER_FUNCTION and HIGH_DENSITY:
                duration = HIGH_POWER_TIME*1000.0
            elif POWER_FUNCTION and LOW_DENSITY:
                duration = LOW_POWER_TIME*1000.0
                
            if not (self.prev_action == -1):
                e = self.utility.get_eccentricity((self.prev_action/MAX_COL),(self.prev_action%MAX_COL),(action/MAX_COL),(action%MAX_COL))
                self.total_time += (duration + 2.7 * e)
            else:
                self.total_time += duration 
            reward = 0.0

        elif action == CLICK_ACTION:
            features = self.current_display[int(self.prev_action/MAX_COL)][int(self.prev_action%MAX_COL)].n_features()
            self.selected_feature = features
            if POWER_FUNCTION:
                reward = POWER_REWARD_TEST[features]/(self.total_time/1000.0)
            else:
                reward = LINEAR_REWARD_TEST[features]/(self.total_time/1000.0)
                
        
        
        if action == CLICK_ACTION:
            done = True
        
        if self.steps >= MAX_STEPS:
            if action < N_IMAGES:
                features = self.current_display[int(action/MAX_COL)][int(action%MAX_COL)].n_features()
            else:
                features = self.current_display[int(self.prev_action/MAX_COL)][int(self.prev_action%MAX_COL)].n_features()
            self.selected_feature = features  
            if POWER_FUNCTION:
                reward = POWER_REWARD_TEST[features]/(self.total_time/1000.0)
            else:
                reward = LINEAR_REWARD_TEST[features]/(self.total_time/1000.0)
            done = True
            
        
        return self.observation.flatten(), self.focus.flatten(), reward, done
    
    def reset(self):	
        self.steps = 0
        self.selected_feature = 0
        self.total_time = 0.00001
        self.prev_action = -1
        self.current_display = self.generator.getNewDisplay()
        self.focus = np.reshape(np.zeros((MAX_ROW,MAX_COL)).flatten(), [N_IMAGES])
        self.observation = np.reshape(np.zeros((MAX_ROW,MAX_COL)).flatten(), [N_IMAGES])

        return self.observation, self.focus
        
        
