# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:40:42 2018

@author: Aditya
"""

class Display(object):
    
    def __init__(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        
        self.feature_1 = int(feature_1)
        self.feature_2 = int(feature_2)
        self.feature_3 = int(feature_3)
        self.feature_4 = int(feature_4)
        self.feature_5 = int(feature_5)

    def n_features(self):
        return (self.feature_1 + self.feature_2 + self.feature_3 + self.feature_4 + self.feature_5)

    def __repr__(self):
        
        return str(self.feature_1 + self.feature_2 + self.feature_3 + 
                self.feature_4 + self.feature_5)
        