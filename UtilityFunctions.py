# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 15:11:40 2018

@author: Aditya
"""
from GlobalConstants import IMAGE_SIZE, GAP_LD, GAP_HD, HIGH_DENSITY, POWER_FUNCTION
from scipy.spatial import distance

class UtilityFunctions(object):
    
    def get_eccentricity(self, fix_x, fix_y, ext_x, ext_y):
        
        x1 = fix_x * (IMAGE_SIZE + (GAP_HD if HIGH_DENSITY else GAP_LD))
        y1 = fix_y * (IMAGE_SIZE + (GAP_HD if HIGH_DENSITY else GAP_LD))
        
        x2 = ext_x * (IMAGE_SIZE + (GAP_HD if HIGH_DENSITY else GAP_LD))
        y2 = ext_y * (IMAGE_SIZE + (GAP_HD if HIGH_DENSITY else GAP_LD))
        
        dst = distance.euclidean([x1,y1],[x2,y2])
        
        return dst
    
