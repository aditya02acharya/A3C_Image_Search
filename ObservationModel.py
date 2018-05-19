# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:59:39 2018

@author: Aditya
"""
from GlobalConstants import MAX_COL, MAX_ROW, IMAGE_SIZE
from UtilityFunctions import UtilityFunctions
import numpy as np
from DisplayGenerator import DisplayGenerator 

class ObservationModel(object):
    
    def __init__(self):
        
        self.utility = UtilityFunctions()
     
    
    def sample(self, display, fixation_loc):
        
        x = fixation_loc / MAX_COL
        y = fixation_loc % MAX_COL
        
        noisy_observation = self.add_feature_noise(display, x, y)
        
        return noisy_observation
        
    
    def add_feature_noise(self, display, fix_x, fix_y):
        
        obs = np.zeros((MAX_ROW, MAX_COL))
        
        for ext_x in range(0, MAX_ROW, 1):
            for ext_y in range(0, MAX_COL, 1):
                e = self.utility.get_eccentricity(fix_x, fix_y, ext_x, ext_y)
                
                mu = 0.1 + (0.1*e) + (0.2*e*e)
                features = 0
                f_size = IMAGE_SIZE/display[ext_x][ext_y].n_features()
                for i in range(display[ext_x][ext_y].n_features()):
                    X = np.random.normal(f_size,0.7*f_size,1)[0]
                    if  (f_size+X) > mu:
                        features += 1
                
                obs[ext_x][ext_y] = features
                
        return obs                    
    

#gen = DisplayGenerator()  

#dis = gen.getNewDisplay()
  
#model = ObservationModel()

#print(model.sample(dis, 14))