# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 13:52:08 2020

@author: Jack Vincent
"""
from math import dist

class Muscle:
    
    def __init__(self, name, F_max, joint, rotation, optimal_fiber_length, 
                 bone1, bone2, origin, insertion, wraps=False, 
                 wrap_transform=(0, 0), wrap_point=(0, 0), min_act=0.05, 
                 max_act=0.95):
        
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
        
        self.wraps = wraps
        self.wrap_transform = wrap_transform
        self.wrap_point = wrap_point
        
        self.length = 0
        
        self.min_act = min_act
        self.max_act = max_act
        
class Musculature:
    
    def __init__(self, muscles, associated_skeleton):
        
        self.muscles = muscles
        self.associated_skeleton = associated_skeleton
    
    # also updates wrap points and lengths
    def update_muscle_endpoints(self):
        for muscle in self.muscles:
            
            for bone in self.associated_skeleton.bones:
                if bone.name == muscle.bone1:
                    bone1 = bone
                elif bone.name == muscle.bone2:
                    bone2 = bone
              
            # can also be thought of as vector form of bone
            x_transform = bone1.endpoint2.coords[0] - bone1.endpoint1.coords[0]
            y_transform = bone1.endpoint2.coords[1] - bone1.endpoint1.coords[1]
            
            x = bone1.endpoint1.coords[0] + (muscle.origin * x_transform)
            y = bone1.endpoint1.coords[1] + (muscle.origin * y_transform)
            
            muscle.endpoint1 = [x, y]
            
            # can also be thought of as vector form of bone
            x_transform = bone2.endpoint2.coords[0] - bone2.endpoint1.coords[0]
            y_transform = bone2.endpoint2.coords[1] - bone2.endpoint1.coords[1]
            
            x = bone2.endpoint1.coords[0] + (muscle.insertion * x_transform)
            y = bone2.endpoint1.coords[1] + (muscle.insertion * y_transform)
            
            muscle.endpoint2 = [x, y]
            
            if muscle.wraps == True:
                for joint in self.associated_skeleton.joints:
                    if joint.name == muscle.joint:
                        joint_loc = joint.location
                
                muscle.wrap_point = [sum(x) for x in zip(joint_loc, muscle.wrap_transform)]
            
            if muscle.wraps == True:
                first_seg = dist(muscle.endpoint1, muscle.wrap_point)
                second_seg = dist(muscle.wrap_point, muscle.endpoint2)
                muscle.length = first_seg + second_seg
            else:
                muscle.length = dist(muscle.endpoint1, muscle.endpoint2)
                
                
                 
            
       