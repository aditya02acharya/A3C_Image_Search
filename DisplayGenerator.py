# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:48:15 2018

@author: Aditya
"""
import numpy as np
from Display import Display

class DisplayGenerator(object):
	
    def __init__(self):
	
        display = []
        
        #Add 5 feature Image.
        imgWithFiveFeatures = Display(True, True, True, True, True)
        display.append(imgWithFiveFeatures)
        
        #Add 4 feature Image.
        imgWithFourFeatures = Display(True, True, True, True, False)
        display.append(imgWithFourFeatures)
        
        #Add 3 feature Image.
        imgWithThreeFeatures_1 = Display(True, True, True, False, False)
        display.append(imgWithThreeFeatures_1)
        imgWithThreeFeatures_2 = Display(True, False, True, False, True)
        display.append(imgWithThreeFeatures_2)
        
        #Add 2 feature Image.
        for i in range(6):
            imgWithTwoFeatures = Display(True, True, False, False, False)
            display.append(imgWithTwoFeatures)
            
        #Add 1 feature Image.
        for i in range(26):
            imgWithOneFeature = Display(True, False, False, False, False)
            display.append(imgWithOneFeature)
        
        self.grid = np.array(display)
        

    def getNewDisplay(self):
        np.random.shuffle(self.grid)
        return self.grid.reshape((6,6))
