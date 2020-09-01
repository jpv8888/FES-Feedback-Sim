# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:52:08 2020

@author: Jack Vincent
"""

class Muscle:
    
    def __init__(self, F_max, joint, rotation):
        
        self.F_max = F_max
        self.joint = joint
        self.rotation = rotation
        
class Musculature:
    
    def __init__(self, muscles):
        
        self.muscles = muscles