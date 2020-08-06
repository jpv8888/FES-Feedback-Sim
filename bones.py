# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 06:21:39 2020

@author: 17049
"""

import math
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


class Endpoint:
    
    def __init__(self, coords, mutable):
        self.coords = coords
        self.mutable = mutable
        

class Bone:
    
    def __init__(self, name, length, mass, mutable1, mutable2):
        self.name = name
        
        # units are MKS
        self.length = length
        self.mass = mass
        
        # calculate moment of inertia about center of mass and joint
        self.I_CoM = (self.mass*(self.length**2))/12
        self.I_joint = self.I_CoM + self.mass*(self.length/2)**2
        self.mutable1 = mutable1
        self.mutable2 = mutable2
        
        # endpoint1 is always endpoint closest toward body
        self.endpoint1 = Endpoint([0, 0], mutable1)
        self.endpoint2 = Endpoint([0, 0], mutable2)
    
    
    CoM = [0, 0]
    
    # calculate center of mass of the bone using the locations of its endpoints
    def calc_CoM(self):
        x = self.endpoint1.coords[0] + (self.endpoint2.coords[0] - self.endpoint1.coords[0])/2
        y = self.endpoint1.coords[1] + (self.endpoint2.coords[1] - self.endpoint1.coords[1])/2
        self.CoM = [x, y]
        
        
class Joint:
    
    def __init__(self, name, bone1, bone2, min_ang, max_ang, default, rotation):
        self.name = name
        
        # bone1 is always bone closest toward body
        self.bone1 = bone1
        self.bone2 = bone2
        
        self.min_ang = min_ang
        self.max_ang = max_ang
        
        # joint's default angle
        self.default = default
        
        # angle defined as angle from bone1 to bone2 starting at 0, units = rads
        self.angle = default
        
        # whether or not bone2 rotates clockwise (cw) or counterclockwise (ccw)
        # relative to bone1
        self.rotation = rotation
    
    location = [0, 0]
        

class Skeleton:
    
    def __init__(self, bones, joints, endpoints_0):
        self.bones = bones
        self.joints = joints
        self.init_endpoints(endpoints_0)
        self.calc_joint_angles()
            
    def init_endpoints(self, endpoints_0):
        for endpoints_bone in endpoints_0:
            for bone in self.bones:
                if bone.name == endpoints_bone:
                    bone.endpoint1.coords = endpoints_0[endpoints_bone][0]
                    bone.endpoint2.coords = endpoints_0[endpoints_bone][1]
                    
    def recalc(self):
        for bone in self.bones:
            bone.calc_CoM()
            bone.calc_I()
    
    def visualize(self):
        segs = []
        
        # anchor for first bone
        anchor_x = self.bones[0].endpoint1.coords[0]
        anchor_y = self.bones[0].endpoint1.coords[1]
        segs.append(((anchor_x, anchor_y + 0.05), (anchor_x, anchor_y - 0.05)))
        
        for bone in self.bones:
            segs.append((tuple(bone.endpoint1.coords), tuple(bone.endpoint2.coords)))
            
        line_segments = LineCollection(segs)
        
        fig, ax = plt.subplots()
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        
        ax.add_collection(line_segments)
        plt.show()
        
    # fix bones so that all endpoints line up, must be called after adjusting 
    # the location of any bone
    def realign_bones(self):
        
        # iterate down the arm (outward from body per convention)
        iterbones = iter(self.bones)
        
        # skip the first bone as its always fixed
        next(iterbones)
        prev_bone_endpoint = self.bones[0].endpoint2.coords
        
        for bone in iterbones:
            if bone.endpoint1.coords != prev_bone_endpoint:
                
                # shift bone such that its proximal endpoint aligns with the
                # distal endpoint of the previous bone while maintaining the
                # location of the current bone's proximal and distal endpoints
                # relative to one another
                bone_vec = [x - y for x, y in zip(bone.endpoint2.coords, bone.endpoint1.coords)]
                bone.endpoint1.coords = prev_bone_endpoint
                bone.endpoint2.coords = [x + y for x, y in zip(bone_vec, bone.endpoint1.coords)]
                
                # also update the location of the relevant joint
                for joint in self.joints:
                    if joint.bone2 == bone.name:
                        joint.location = bone.endpoint1.coords
            
            prev_bone_endpoint = bone.endpoint2.coords
            
    # calculate and set joint angles using bone endpoints
    # cosine of angle between two vectors is defined as their dot product 
    # divided by the product of the magnitudes of the two vectors
    def calc_joint_angles(self):
        for joint in self.joints:
            for bone in self.bones:
                if joint.bone1 == bone.name:
                    joint_loc = bone.endpoint2.coords
                    joint_endpoint1 = bone.endpoint1.coords
                elif joint.bone2 == bone.name:
                    joint_endpoint2 = bone.endpoint2.coords
            
            joint.location = joint_loc
            
            vec1 = [x - y for x, y in zip(joint_endpoint1, joint_loc)]
            vec2 = [x - y for x, y in zip(joint_endpoint2, joint_loc)]
            
            vec1_mag = (vec1[0]**2 + vec1[1]**2)**0.5
            vec2_mag = (vec2[0]**2 + vec2[1]**2)**0.5
            
            dot_prod = vec1[0]*vec2[0] + vec1[1]*vec2[1]
            
            cos_alpha = dot_prod/(vec1_mag*vec2_mag)
            
            joint.angle = math.acos(cos_alpha)
    
    # calculate and set bone endpoints using joint angles       
    def calc_bone_endpoints(self):
        current_bone_endpoint1 = [0, 0]
        for bone in self.bones:
            prev_bone_endpoint1 = current_bone_endpoint1
            current_bone_endpoint1 = bone.endpoint1.coords
            if bone.endpoint2.mutable == True:
                for joint in self.joints:
                    if joint.bone2 == bone.name:
                        
                         joint_loc = bone.endpoint1.coords
                         joint_endpoint1 = prev_bone_endpoint1
                         
                         vec1 = [x - y for x, y in zip(joint_endpoint1, joint_loc)]
                         vec1_mag = (vec1[0]**2 + vec1[1]**2)**0.5
                         vec1_unit = []
                         for coord in vec1:
                             vec1_unit.append(coord/vec1_mag)
                         if joint.rotation == 'ccw':
                             x = vec1_unit[0]*math.cos(joint.angle) - vec1_unit[1]*math.sin(joint.angle)
                             y = vec1_unit[0]*math.sin(joint.angle) + vec1_unit[1]*math.cos(joint.angle)
                         elif joint.rotation == 'cw':
                             x = vec1_unit[0]*math.cos(joint.angle) + vec1_unit[1]*math.sin(joint.angle)
                             y = vec1_unit[0]*-math.sin(joint.angle) + vec1_unit[1]*math.cos(joint.angle)
                         vec2_unit = [x, y]
                         
                         vec2 = []
                         for coord in vec2_unit:
                             vec2.append(coord * bone.length)
                            
                         bone.endpoint2.coords = [x + y for x, y in zip(vec2, joint_loc)]
                         self.realign_bones()
                         
    def print_data(self):
         for bone in self.bones:
             print(bone.name)
             print(bone.endpoint1.coords)
             print(bone.endpoint2.coords)
         for joint in self.joints:
             print(joint.name)
             print(joint.location)
             print(joint.angle)
            
            
            
