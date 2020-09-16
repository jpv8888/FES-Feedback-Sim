# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:52:08 2020

@author: Jack Vincent
"""

class Muscle:
    
    def __init__(self, name, F_max, joint, rotation, optimal_fiber_length, 
                 bone1, bone2, origin, insertion, wraps = False):
        
        self.name = name
        self.F_max = F_max
        self.joint = joint
        self.rotation = rotation
        self.optimal_fiber_length = optimal_fiber_length
        self.bone1 = bone1
        self.bone2 = bone2
        
        # relative to bone1, traveling from endpoint1 to endpoint2, fraction 
        # of total length of bone traveled
        self.origin = origin
        
        # relative to bone2, traveling from endpoint1 to endpoint2, fraction 
        # of total length of bone traveled
        self.insertion = insertion
        
        # current location of muscle origin and insertion in xy space, needs 
        # to be initialized, impossible to know without knowing setup of 
        # skeleton
        self.endpoint1 = [0, 0]
        self.endpoint2 = [0, 0]
        
class Musculature:
    
    def __init__(self, muscles):
        
        self.muscles = muscles