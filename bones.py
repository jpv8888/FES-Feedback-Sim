# -*- coding: utf-8 -*-
"""
Classes and function comprising all skeletal elements of the model
@author: Jack Vincent
"""

import alphashape
from descartes import PolygonPatch
import itertools
import math
from math import dist, radians
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


# only really a class so that bone endpoints can be checked for mutability 
class Endpoint:
    
    def __init__(self, coords, mutable):
        self.coords = coords
        self.mutable = mutable
        
# first of two chief components making up a skeleton
class Bone:
    
    def __init__(self, name, length, mass, mutable1, mutable2):
        self.name = name
        
        # units are MKS
        self.length = length
        self.mass = mass
        
        # calculate moment of inertia about center of mass and joint, 
        # specifically joint immediately preceding bone, as bone's moment of 
        # inertia contribution to that joint will be constant regardless of 
        # its exact orientation
        self.I_CoM = (self.mass*(self.length**2))/12
        self.I_joint = self.I_CoM + self.mass*(self.length/2)**2
        
        # mutability of endpoints
        self.mutable1 = mutable1
        self.mutable2 = mutable2
        
        # endpoint1 is always endpoint closest toward body
        self.endpoint1 = Endpoint([0, 0], mutable1)
        self.endpoint2 = Endpoint([0, 0], mutable2)
    
    # center of mass: bone endpoints need to be set before can be calculated
    CoM = [0, 0]
    
    # calculate the bone's center of mass using the locations of its endpoints
    def calc_CoM(self):
        x = self.endpoint1.coords[0] + (self.endpoint2.coords[0] - self.endpoint1.coords[0])/2
        y = self.endpoint1.coords[1] + (self.endpoint2.coords[1] - self.endpoint1.coords[1])/2
        self.CoM = [x, y]
        
# second of two chief components making up a skeleton      
class Joint:
    
    def __init__(self, name, bone1, bone2, min_ang, max_ang, default, rotation):
        self.name = name
        
        # bone1 is always bone closest toward body
        self.bone1 = bone1
        self.bone2 = bone2
        
        # minimum and maximum allowed joint angles, units = degrees
        self.min_ang = min_ang
        self.max_ang = max_ang
        
        # joint's default angle, units = degrees
        self.default = default
        
        # angle is defined as angle from bone1 to bone2 starting at 0; 
        # units = rads, why? min_ang, max_ang, and default are all prescribed
        # by the user in advance, angle is really the only value used in model
        # operations
        self.angle = default
        
        # whether or not bone2 rotates clockwise (cw) or counterclockwise 
        # (ccw) relative to bone1
        self.rotation = rotation
        
    
    # bone endpoints need to be set before can be calculated
    location = [0, 0]
    
    # moment of inertia for rotating this joint
    I = 0
        
# comprised of bones and joints
class Skeleton:
    
    def __init__(self, bones, joints, endpoints_0):
        
        # initial endpoints must be provided to set the skeleton at start
        self.bones = bones
        self.joints = joints
        self.init_endpoints(endpoints_0)
        self.calc_alpha_shape()
        
    # sets bone endpoint locations base on initial input endpoint locations, 
    # runs on object instantiation     
    def init_endpoints(self, endpoints_0):
        for endpoints_bone in endpoints_0:
            for bone in self.bones:
                if bone.name == endpoints_bone:
                    bone.endpoint1.coords = endpoints_0[endpoints_bone][0]
                    bone.endpoint2.coords = endpoints_0[endpoints_bone][1]
                    
        # adjust joint angles accordingly
        self.calc_joint_angles()
    
    # runs on object instantiation, explores endpoint solution space and fits
    # alpha shape to boundaries
    def calc_alpha_shape(self):

        # degrees, precision with which to comb through possible endpoint locations
        angle_step = 1
        
        # will eventually hold all possible joint angle combinations
        angle_space = []
        
        # outputs list of lists, where each list corresponds to all possible values of
        # a joint angle
        for joint in self.joints:
            angle_space.append(np.arange(radians(joint.min_ang), 
                                         radians(joint.max_ang) + radians(angle_step), 
                                         radians(angle_step)))
            
        # converts angle_space list to an iterable of every possible combination of 
        # joint angles
        angle_space = itertools.product(*angle_space)
           
        # records every hand endpoint associated with the previously generated angles
        possible_endpoints = []
        for joint_angles in angle_space:
            self.write_joint_angles(joint_angles)
            possible_endpoints.append(self.bones[-1].endpoint2.coords)
            
        self.possible_endpoints = possible_endpoints
        
        # optimal alpha can be calculated automatically, but doing this results in a 
        # very long run time; user-selected alpha should be big enough to tightly wrap
        # around the solution space but low enough to not exclude points
        alpha = 7
        
        # generate and plot solution space alpha shape
        self.alpha_shape = alphashape.alphashape(possible_endpoints, alpha)
        
        # axis limits
        left = min(endpoint[0] for endpoint in possible_endpoints)
        right = max(endpoint[0] for endpoint in possible_endpoints)
        top = max(endpoint[1] for endpoint in possible_endpoints)
        bottom = min(endpoint[1] for endpoint in possible_endpoints)
        
        # stored for all future visualizations
        self.x_lim = [left - 0.1, right + 0.1]
        self.y_lim = [bottom - 0.1, top + 0.1]
       
    # update moment of inertia at each joint
    def calc_I(self):
        
        add_subsequent_bones = False
        for joint in self.joints:
            for bone in self.bones:
                bone.calc_CoM()
                if bone.name == joint.bone2:
                    I = bone.I_joint
                    add_subsequent_bones = True
                elif add_subsequent_bones == True:
                    bone_dist = dist(bone.CoM, joint.location)
                    I_bone = bone.I_CoM + bone.mass*(bone_dist)**2
                    I += I_bone
            joint.I = I
            add_subsequent_bones = False
                
        
    # view hand endpoint solution space and associated alpha shape
    def plot_solution_space(self):
        
        # figure formatting
        fig, ax = plt.subplots()
        ax.axis('square')
        ax.set_xlim(self.x_lim[0], self.x_lim[1])
        ax.set_ylim(self.y_lim[0], self.y_lim[1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Possible Hand Endpoint Locations')
        
        # add the possible hand endpoints
        ax.scatter(*zip(*self.possible_endpoints), 0.1)
        
        # add a circle to show where the shoulder is
        circle = patches.Circle((self.joints[0].location[0], self.joints[0].location[1]), 
                                0.015, fill=True)
        ax.add_patch(circle)
        plt.annotate('shoulder', 
                     (self.joints[0].location[0], self.joints[0].location[1]))
        
        # add the alpha shape
        ax.add_patch(PolygonPatch(self.alpha_shape, alpha=0.2)) 
    
    # calculate and set joint angles using bone endpoints, also updates joint
    # locations, cosine of angle between two vectors is defined as their dot 
    # product divided by the product of the magnitudes of the two vectors,
    # runs on object instantiation 
    def calc_joint_angles(self):
        
        # iterate through each joint
        for joint in self.joints:
            
            # identify the bones associated with this joint
            for bone in self.bones:
                if joint.bone1 == bone.name:
                    joint_loc = bone.endpoint2.coords
                    joint_endpoint1 = bone.endpoint1.coords
                elif joint.bone2 == bone.name:
                    joint_endpoint2 = bone.endpoint2.coords
            
            # update the joint location
            joint.location = joint_loc
            
            # calculate vectors describing bones enclosing joint
            vec1 = [x - y for x, y in zip(joint_endpoint1, joint_loc)]
            vec2 = [x - y for x, y in zip(joint_endpoint2, joint_loc)]
            
            # calculate those vectors' magnitudes
            vec1_mag = (vec1[0]**2 + vec1[1]**2)**0.5
            vec2_mag = (vec2[0]**2 + vec2[1]**2)**0.5
            
            # calculate dot product between those vectors
            dot_prod = vec1[0]*vec2[0] + vec1[1]*vec2[1]
            
            # calculate cosine of joint angle
            cos_alpha = dot_prod/(vec1_mag*vec2_mag)
            
            # calculate and write joint angle
            joint.angle = math.acos(cos_alpha)
            
    """
    this doesn't take into account joint rotation direction, but it absolutely
    should, the values it's producing may not be correct
    """
    
    # calculate and set bone endpoints using joint angles, also updates joint
    # locations due to calling of realign_bones()       
    def calc_bone_endpoints(self):
        
        # start with the first bone
        current_bone_endpoint1 = self.bones[0].endpoint1.coords
        
        # iterate down the arm (outward from body per convention)
        iterbones = iter(self.bones)
        
        # skip the first bone as its always fixed
        next(iterbones)
        
        for bone in iterbones:
            prev_bone_endpoint1 = current_bone_endpoint1
            current_bone_endpoint1 = bone.endpoint1.coords
            
            # check if the current bone's second endpoint can be moved
            if bone.endpoint2.mutable == True:
                
                # identify joint associated with the current movable bone
                for joint in self.joints:
                    if joint.bone2 == bone.name:
                        
                         # redefining some stuff relative to the joint of 
                         # interest
                         joint_loc = bone.endpoint1.coords
                         joint_endpoint1 = prev_bone_endpoint1
                         
                         # vector representing previous bone relative to joint
                         vec1 = [x - y for x, y in zip(joint_endpoint1, joint_loc)]
                         
                         # magnitude of that vector
                         vec1_mag = (vec1[0]**2 + vec1[1]**2)**0.5
                         
                         # that vector converted to a unit vector
                         vec1_unit = []
                         for coord in vec1:
                             vec1_unit.append(coord/vec1_mag)
                             
                        # counterclockwise or clockwise (depending on rotation 
                        # direction of joint) rotation matrix applied to that 
                        # unit vector, output is now unit vector 
                        # representation of joint's second bone
                         if joint.rotation == 'ccw':
                             x = vec1_unit[0]*math.cos(joint.angle) - vec1_unit[1]*math.sin(joint.angle)
                             y = vec1_unit[0]*math.sin(joint.angle) + vec1_unit[1]*math.cos(joint.angle)
                         elif joint.rotation == 'cw':
                             x = vec1_unit[0]*math.cos(joint.angle) + vec1_unit[1]*math.sin(joint.angle)
                             y = vec1_unit[0]*-math.sin(joint.angle) + vec1_unit[1]*math.cos(joint.angle)
                         vec2_unit = [x, y]
                         
                         # scale that unit vector by second bone's length
                         vec2 = []
                         for coord in vec2_unit:
                             vec2.append(coord * bone.length)
                             
                         # adjust second bone's endpoints according to joint 
                         # location and vector that now represents it rotated 
                         # correctly about the joint
                         bone.endpoint2.coords = [x + y for x, y in zip(vec2, joint_loc)]
                         
                         # must now realign bones, as any bones that might 
                         # have been connected to the bone we just adjusted 
                         # are now in the wrong place
                         self.realign_bones()
                         
    # fix bones so that all endpoints line up, must be called after adjusting 
    # the location of any bone
    def realign_bones(self):
        
        # iterate down the arm (outward from body per convention)
        iterbones = iter(self.bones)
        
        # skip the first bone as its always fixed
        next(iterbones)
        
        # end-endpoint of the previous bone, i.e. endpoint 2
        prev_bone_endpoint = self.bones[0].endpoint2.coords
        
        # check other, non-fixed bones to see if they are lining up with their 
        # previous bones
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
            
            # this bone is now the previous bone
            prev_bone_endpoint = bone.endpoint2.coords
    
    # can be used to obtain all current joint angles of the skeleton 
    def return_joint_angles(self):
        
        joint_angles = []
        
        for joint in self.joints:
            joint_angles.append(joint.angle)
            
        joint_angles = tuple(joint_angles)
        
        return joint_angles
    
    # set user or script defined joint angles and update all necessary bone 
    # and joint parameters                      
    def write_joint_angles(self, joint_angles):
        
        for i, joint_angle in enumerate(joint_angles):
            self.joints[i].angle = joint_angle
        
        self.calc_bone_endpoints()
        
    # update CoM and moments of inertia for each bone      
    def recalc(self):
    
        for bone in self.bones:
            bone.calc_CoM()
           

    """
    no use at the moment
    """