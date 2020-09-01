# -*- coding: utf-8 -*-
"""
Classes and functions for generating a completed model and coordinating 
actions that require crosstalk between skeletal and muscular information
@author: Jack Vincent
"""

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pickle

from bones import Bone, Joint, Skeleton
from muscles import Muscle, Musculature

# writes and dumps a fresh model in the working directory
def init_model(name):
    
    # name for this particular model instance
    name = name
    
    # for bone lengths:
    # Pan N. Length of long bones and their proportion to body height in 
    # hindus. J Anat. 1924;58:374-378. 
    # https://pubmed.ncbi.nlm.nih.gov/17104032 
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1249729/.
    
    # for segment masses, mass information is stored in bones, but is assumed
    # to also account for mass of muscle, connective tissue, etc.:
    # DA Winter, Biomechanics and Motor Control of Human Movement, 3rd edition 
    # (John Wiley & Sons 2005)
    
    # units are MKS
    height = 1.7
    mass = 65
    
    # "scapula" is a major abstraction, mass is irrelevant, and length is
    # irrelevant beyond giving sufficient room for muscle attachment sites, so 
    # these dimensions are mostly arbitrary 
    scapula = Bone('scapula', 0.05*height, 0.01*mass, False, False)
    
    # generate other bones
    humerus = Bone('humerus', 0.188*height, 0.028*mass, False, True)
    radioulna  = Bone('radioulna', 0.164*height, 0.022*mass, True, True)
    
    # initial bone endpoints, shoulder joint is always fixed and defined as 
    # the origin 
    endpoints_0 = {'scapula': ([scapula.length, 0], [0, 0]),
                   'humerus': ([0, 0], [0, -humerus.length]), 
                   'radioulna': ([0, -humerus.length], 
                                 [0, -humerus.length-radioulna.length])}
    
    # generate joints
    shoulder = Joint('shoulder', 'scapula', 'humerus', 60, 210, 90, 'cw')
    elbow = Joint('elbow', 'humerus', 'radioulna', 30, 180, 180, 'ccw')
    
    # generate fresh, completed skeleton
    skeleton = Skeleton([scapula, humerus, radioulna], [shoulder, elbow], 
                        endpoints_0)
    
    # dump model
    with open(name, "wb") as fp:
        pickle.dump(Model(name, skeleton), fp)
        
# writes and dumps a fresh experiment in the working directory
def init_experiment(name, f_s):
    
    # dump experiment
    with open(name, "wb") as fp:
        pickle.dump(Experiment(name, f_s), fp)
        
# load in a model or experiment from the working directory
def load(name):
    
    # read in model or experiment
    with open(name, "rb") as fp:   
        return(pickle.load(fp))

# model object composed of skeleton and musculature 
class Model:
    
    def __init__(self, name, skeleton):
        
        self.name = name
        self.skeleton = skeleton
    
    # store self in working directory using pickle
    def dump(self):
        with open(self.name, "wb") as fp:
            pickle.dump(self, fp)
            
    # generate still image of skeleton, primarily for debugging purposes                     
    def visualize(self):
        
        # will hold all bones to be plotted as line segments
        segs = []
        
        # anchor for first bone
        anchor_x = self.skeleton.bones[0].endpoint1.coords[0]
        anchor_y = self.skeleton.bones[0].endpoint1.coords[1]
        segs.append(((anchor_x, anchor_y + 0.05), (anchor_x, anchor_y - 0.05)))
        
        # add all bones to the collection of segments to be plotted
        for bone in self.skeleton.bones:
            segs.append((tuple(bone.endpoint1.coords), tuple(bone.endpoint2.coords)))
            
        line_segments = LineCollection(segs)
        
        # will hold all joints and hand to be plotted as circles
        circles = []
        
        # add circles representing each joint
        for joint in self.skeleton.joints:
            circle = patches.Circle((joint.location[0], joint.location[1]), 
                                    0.01, fill=True)
            circles.append(circle)
            
        # add circle representing hand
        circle = patches.Circle((self.skeleton.bones[-1].endpoint2.coords[0], 
                                 self.skeleton.bones[-1].endpoint2.coords[1]), 0.015, 
                                fill=True)
        circles.append(circle)
        
        circles_collection = PatchCollection(circles)
            
        # initialize figure
        fig, ax = plt.subplots()
        
        # circles look like ovals if you don't include this 
        ax.axis('square')
        
        # formatting
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Current Skeleton')
        ax.set_xlim(self.skeleton.x_lim[0], self.skeleton.x_lim[1])
        ax.set_ylim(self.skeleton.y_lim[0], self.skeleton.y_lim[1])
        
        # add all visualization elements
        ax.add_collection(line_segments)
        ax.add_collection(circles_collection)
        
    # animate data described by joint angles over time; first create frame 1 
    # and its associated line and circle collections, then use an animation 
    # function with FuncAnimation to create all subsequent frames
    def animate(self, experiment, save=False):
        
        # sampling frequency
        interval = 1/experiment.f_s
        
        # data formatting
        joint_angles_pre_zip = []
        for joint_data in experiment.joints:
            joint_angles_pre_zip.append(joint_data.angle)
            
        joint_angles = [list(a) for a in zip(*joint_angles_pre_zip)]
        
        # initialize figure
        fig, ax = plt.subplots()
        
        # circles look like ovals if you don't include this 
        ax.axis('square')
        
        # formatting
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Animated Skeleton')
        ax.set_xlim(self.skeleton.x_lim[0], self.skeleton.x_lim[1])
        ax.set_ylim(self.skeleton.y_lim[0], self.skeleton.y_lim[1])
        
        # set skeleton for first frame
        self.skeleton.write_joint_angles(joint_angles[0])
        
        # will hold all bones to be plotted as line segments
        segs = []
        
        # anchor for first bone
        anchor_x = self.skeleton.bones[0].endpoint1.coords[0]
        anchor_y = self.skeleton.bones[0].endpoint1.coords[1]
        segs.append(((anchor_x, anchor_y + 0.05), (anchor_x, anchor_y - 0.05)))
        
        # add all bones to the collection of segments to be plotted
        for bone in self.skeleton.bones:
            segs.append((tuple(bone.endpoint1.coords), tuple(bone.endpoint2.coords)))
            
        line_segments = LineCollection(segs)
        
        # will hold all joints and hand to be plotted as circles
        circles = []
        
        # add circles representing each joint
        for joint in self.skeleton.joints:
            circle = patches.Circle((joint.location[0], joint.location[1]), 
                                    0.01, fill=True)
            circles.append(circle)
            
        # add circle representing hand
        circle = patches.Circle((self.skeleton.bones[-1].endpoint2.coords[0], 
                                 self.skeleton.bones[-1].endpoint2.coords[1]), 
                                0.015, fill=True)
        circles.append(circle)
        
        circles_collection = PatchCollection(circles)
        
        # add visualization elements for first frame
        ax.add_collection(line_segments)
        ax.add_collection(circles_collection)
        
        # other stuff that needs to be passed to our animation function
        fargs = self, joint_angles
        
        # function that will be called for each frame of animation
        def func(frame, *fargs):
            
            # set skeleton
            self.skeleton.write_joint_angles(joint_angles[frame])
            
            # will hold all bones to be plotted as line segments
            segs = []
        
            # anchor for first bone
            anchor_x = self.skeleton.bones[0].endpoint1.coords[0]
            anchor_y = self.skeleton.bones[0].endpoint1.coords[1]
            segs.append(((anchor_x, anchor_y + 0.05), (anchor_x, anchor_y - 0.05)))
            
            # add all bones to the collection of segments to be plotted
            for bone in self.skeleton.bones:
                segs.append((tuple(bone.endpoint1.coords), tuple(bone.endpoint2.coords)))
                
           
            # will hold all joints and hand to be plotted as circles
            circles = []
            
            # add circles representing each joint
            for joint in self.skeleton.joints:
                circle = patches.Circle((joint.location[0], joint.location[1]), 
                                        0.01, fill=True)
                circles.append(circle)
                
            # add circle representing hand
            circle = patches.Circle((self.skeleton.bones[-1].endpoint2.coords[0], 
                                     self.skeleton.bones[-1].endpoint2.coords[1]), 
                                    0.015, fill=True)
            circles.append(circle)
            
            # update and plot line and circle collections
            line_segments.set_paths(segs)
            circles_collection.set_paths(circles)
            
        # animate each frame using animation function
        anim = FuncAnimation(fig, func, frames=len(joint_angles), 
                             interval=interval)
        
        if save == True:
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=2000)
            anim.save('test.mp4', writer=writer)
        
        return anim
       
    # necessary to be able to pickle this object
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    
# experiment object, keeps track of data through various processing scripts
class Experiment:
    
    def __init__(self, name, f_s):
        
        self.f_s = f_s
        self.T = 1/f_s
        self.name = name
        self.t = []
        self.endpoints = []
        self.joints = []
        
    # store self in working directory using pickle
    def dump(self):
        with open(self.name, "wb") as fp:
            pickle.dump(self, fp)
            
    def plot(self, data):
        if data == 'angle':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('Angle (rad)')
            ax.set_title(self.name + ' Joint Angles')
            
            labels = []
            for joint_data in self.joints:
                plt.plot(self.t, joint_data.angle)
                labels.append(joint_data.name)
                
            ax.legend(labels)
            
        elif data == 'velocity':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('v (rad/s)')
            ax.set_title(self.name + ' Joint Velocities')
            
            labels = []
            for joint_data in self.joints:
                plt.plot(self.t, joint_data.velocity)
                labels.append(joint_data.name)
                
            ax.legend(labels)
            
        elif data == 'acceleration':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('a (rad/s^2)')
            ax.set_title(self.name + ' Joint Accelerations')
            
            labels = []
            for joint_data in self.joints:
                plt.plot(self.t, joint_data.acceleration)
                labels.append(joint_data.name)
                
            ax.legend(labels)
            
        elif data == 'torque':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('tau (N * m)')
            ax.set_title(self.name + ' Joint Torques')
            
            labels = []
            for joint_data in self.joints:
                plt.plot(self.t, joint_data.torque)
                labels.append(joint_data.name)
                
            ax.legend(labels)
        
    # necessary to be able to pickle this object
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

class Simulation:

    def __init__(self, name, model, f_s, duration):
        
        self.name = name
        self.model = model
        self.f_s = f_s
        self.T = 1/f_s
        self.joints = []
        self.t = np.linspace(0, duration, int(duration/self.T))
        self.t = list(self.t)
        self.t = [round(elem, 2) for elem in self.t]
        self.duration = duration
        
        for joint in self.model.skeleton.joints:
            self.joints.append(JointData(joint.name))
            
        for i, joint_data in enumerate(self.joints):
            ang = self.model.skeleton.joints[i].angle
            joint_data.angle = [ang, ang]
            joint_data.velocity = [0, 0]
            joint_data.acceleration = [0, 0]
            joint_data.torque = [0, 0]
            
    def plot(self, data):
        if data == 'angle':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('Angle (rad)')
            ax.set_title(self.name + ' Joint Angles')
            
            labels = []
            for joint_data in self.joints:
                plt.plot(self.t, joint_data.angle)
                labels.append(joint_data.name)
                
            ax.legend(labels)
            
        elif data == 'velocity':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('v (rad/s)')
            ax.set_title(self.name + ' Joint Velocities')
            
            labels = []
            for joint_data in self.joints:
                plt.plot(self.t, joint_data.velocity)
                labels.append(joint_data.name)
                
            ax.legend(labels)
            
        elif data == 'acceleration':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('a (rad/s^2)')
            ax.set_title(self.name + ' Joint Accelerations')
            
            labels = []
            for joint_data in self.joints:
                plt.plot(self.t, joint_data.acceleration)
                labels.append(joint_data.name)
                
            ax.legend(labels)
            
        elif data == 'torque':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('tau (N * m)')
            ax.set_title(self.name + ' Joint Torques')
            
            labels = []
            for joint_data in self.joints:
                plt.plot(self.t, joint_data.torque)
                labels.append(joint_data.name)
                
            ax.legend(labels)
            
    # store self in working directory using pickle
    def dump(self):
        with open(self.name, "wb") as fp:
            pickle.dump(self, fp)
              
    # necessary to be able to pickle this object
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
            
# tracking data related to a single joint through an experiment
class JointData:
    
    def __init__(self, name):
    
        self.name = name
        self.angle = []
        self.velocity = []
        self.acceleration = []
        self.torque = []
        

        

