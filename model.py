# -*- coding: utf-8 -*-
"""
Classes and functions for generating a completed model and coordinating 
actions that require crosstalk between skeletal and muscular information
@author: Jack Vincent
"""
from autograd import grad

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.misc import derivative
from scipy.interpolate import UnivariateSpline

from bones import Bone, Joint, Skeleton
from muscles import Muscle, BiarticularMuscle, Musculature

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
    radioulna = Bone('radioulna', 0.164*height, 0.022*mass, True, True)
    radioulna = Bone('radioulna', 0.164*height, 0.022*mass, True, True, 
                     point_mass=(4.5, 'endpoint_2'))
    
    # initial bone endpoints, shoulder joint is always fixed and defined as 
    # the origin 
    endpoints_0 = {'scapula': ([scapula.length, 0], [0, 0]),
                   'humerus': ([0, 0], [0, -humerus.length]), 
                   'radioulna': ([0, -humerus.length], 
                                 [0, -humerus.length-radioulna.length])}
    
    # generate joints
    shoulder = Joint('shoulder', 'scapula', 'humerus', 60, 210, 90, 'cw', 0.02)
    elbow = Joint('elbow', 'humerus', 'radioulna', 30, 180, 180, 'ccw', 0.02)
    
    # generate fresh, completed skeleton
    skeleton = Skeleton([scapula, humerus, radioulna], [shoulder, elbow], 
                        endpoints_0)
    
    ant_delt = Muscle('anterior_deltoid', 2500, 'shoulder', 'cw', 0.19, 
                      'scapula', 'humerus', 0.7, 0.4, 0.02, wraps=True, 
                      wrap_transform=(-0.015, 0.015))
    lat_dorsi = Muscle('latissimus_dorsi', 900, 'shoulder', 'ccw', 0.11, 
                       'scapula', 'humerus', 0.1, 0.2, 0.03, wraps=True, 
                       wrap_transform=(0.015, -0.015))
    
    # for simplicity, I'm assuming biarticular muscles have the same moment 
    # arm at each joint
    biceps = BiarticularMuscle('biceps', 1000, ['shoulder', 'elbow'], 
                               ['cw', 'cw'], 0.39, 'scapula', 'radioulna', 0.8, 
                               0.21, 0.02, wraps=True, 
                               wrap_transform=[(-0.015, 0.015), (0.005, 0.015)],
                               wrap_point=[(0, 0), (0, 0)])
    tri_short = Muscle('tri_short', 1200, 'elbow', 'ccw', 0.32, 'humerus', 
                     'radioulna', 0.5, 0.1, 0.02, wraps=True, 
                     wrap_transform=(0.005, -0.015))
    tri_long = BiarticularMuscle('tri_long', 800, ['shoulder', 'elbow'], 
                                 ['ccw', 'ccw'], 0.43, 'scapula', 'radioulna', 
                                 0.1, 0.1, 0.02, wraps=True, 
                                 wrap_transform=[(0.015, -0.015), (0.005, -0.015)],
                                 wrap_point=[(0, 0), (0, 0)])
    brachiorad = Muscle('brachioradialis', 300, 'elbow', 'cw', 0.28, 'humerus', 
                        'radioulna', 0.8, 0.6, 0.06)
    brachialis = Muscle('brachialis', 1000, 'elbow', 'cw', 0.28, 'humerus', 
                        'radioulna', 0.5, 0.1, 0.02)
     
    musculature = Musculature([ant_delt, lat_dorsi, biceps, tri_short, 
                               tri_long, brachiorad, brachialis], skeleton)
    
    current_model = Model(name, skeleton, musculature)
    current_model.musculature.update_muscle_endpoints()
    
    # dump model
    with open(name, "wb") as fp:
        pickle.dump(current_model, fp)
        
# writes and dumps a fresh experiment in the working directory
def init_experiment(name, f_s, associated_model):
    
    # dump experiment
    with open(name, "wb") as fp:
        pickle.dump(Experiment(name, f_s, associated_model), fp)
        
# load in a model or experiment from the working directory
def load(name):
    
    # read in model or experiment
    with open(name, "rb") as fp:   
        return(pickle.load(fp))

# model object composed of skeleton and musculature 
class Model:
    
    def __init__(self, name, skeleton, musculature):
        
        self.name = name
        self.skeleton = skeleton
        self.musculature = musculature
    
    # store self in working directory using pickle
    def dump(self):
        with open(self.name, "wb") as fp:
            pickle.dump(self, fp)
            
    # generate still image of skeleton, primarily for debugging purposes                     
    def visualize(self):
        
        # will hold all bones to be plotted as line segments
        segs = []
        
        # point masses
        markers = []
        
        # anchor for first bone
        anchor_x = self.skeleton.bones[0].endpoint1.coords[0]
        anchor_y = self.skeleton.bones[0].endpoint1.coords[1]
        segs.append(((anchor_x, anchor_y + 0.05), (anchor_x, anchor_y - 0.05)))
        
        # add all bones to the collection of segments to be plotted
        for bone in self.skeleton.bones:
            segs.append((tuple(bone.endpoint1.coords), tuple(bone.endpoint2.coords)))
            if bone.point_mass[0] != 0:
                markers.append([bone.point_mass_loc, bone.point_mass[0]])
            
        line_segments = LineCollection(segs, color='g', linewidth=2, zorder=2)
        
        muscle_segs = []
        
        # add all muscles to a collection of segments to be plotted
        for muscle in self.musculature.muscles:
            if isinstance(muscle, BiarticularMuscle):
                muscle_segs.append((tuple(muscle.endpoint1), 
                                    tuple(muscle.wrap_point[0]), 
                                    tuple(muscle.wrap_point[1]),
                                    tuple(muscle.endpoint2)))
            else:
                
                if muscle.wraps == True:
                    muscle_segs.append((tuple(muscle.endpoint1), 
                                        tuple(muscle.wrap_point), 
                                        tuple(muscle.endpoint2)))
                else:
                    muscle_segs.append((tuple(muscle.endpoint1), 
                                        tuple(muscle.endpoint2)))
                
        muscle_lines = LineCollection(muscle_segs, linewidth=2, color='r',
                                      zorder=1)
        
        # will hold all joints and hand to be plotted as circles
        circles = []
        
        # add circles representing each joint
        for joint in self.skeleton.joints:
            circle = patches.Circle((joint.location[0], joint.location[1]), 
                                    joint.diameter/2, fill=True)
            circles.append(circle)
            
        # add circle representing hand
        circle = patches.Circle((self.skeleton.bones[-1].endpoint2.coords[0], 
                                 self.skeleton.bones[-1].endpoint2.coords[1]), 
                                0.015, fill=True)
        circles.append(circle)
        
        circles_collection = PatchCollection(circles, color='k', zorder=3)
            
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
        ax.add_collection(muscle_lines)
        ax.add_collection(circles_collection)
        for marker in markers:
            ax.plot(marker[0][0], marker[0][1], marker='s', color='purple',
                    markersize=marker[1], zorder=4)
        
    # animate data described by joint angles over time; first create frame 1 
    # and its associated line and circle collections, then use an animation 
    # function with FuncAnimation to create all subsequent frames
    def animate(self, data, save=False, muscle_tracker=None):
        
        # finished video fps
        video_fps = 60
        
        # data formatting
        joint_angles_pre_zip = []
        for joint_data in data.joints:
            joint_angles_pre_zip.append(joint_data.angle)
            
        joint_angles = [list(a) for a in zip(*joint_angles_pre_zip)]
        
        # constants for converting data fps to video fps
        data_fps = data.f_s
        num_data_frames = len(joint_angles)
        vid_length = num_data_frames/data_fps
        num_video_frames = round(vid_length*video_fps)
        
        # initialize figure
        fig, ax = plt.subplots()
        
        # circles look like ovals if you don't include this 
        ax.axis('square')
        
        # formatting
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Animated Model')
        ax.set_xlim(self.skeleton.x_lim[0], self.skeleton.x_lim[1])
        ax.set_ylim(self.skeleton.y_lim[0], self.skeleton.y_lim[1])
        
        # set skeleton for first frame
        self.skeleton.write_joint_angles(joint_angles[0])
        self.musculature.update_muscle_endpoints()
        
        # will hold all bones to be plotted as line segments
        segs = []
        
        markers = []
        
        # anchor for first bone
        anchor_x = self.skeleton.bones[0].endpoint1.coords[0]
        anchor_y = self.skeleton.bones[0].endpoint1.coords[1]
        segs.append(((anchor_x, anchor_y + 0.05), (anchor_x, anchor_y - 0.05)))
        
        # add all bones to the collection of segments to be plotted
        for bone in self.skeleton.bones:
            segs.append((tuple(bone.endpoint1.coords), tuple(bone.endpoint2.coords)))
            if bone.point_mass[0] != 0:
                markers.append([bone.point_mass_loc, bone.point_mass[0]])
            
        line_segments = LineCollection(segs, color='g', linewidth=2, zorder=2)
        
        muscle_segs = []
        
       # add all muscles to a collection of segments to be plotted
        for muscle in self.musculature.muscles:
            if isinstance(muscle, BiarticularMuscle):
                muscle_segs.append((tuple(muscle.endpoint1), 
                                    tuple(muscle.wrap_point[0]), 
                                    tuple(muscle.wrap_point[1]),
                                    tuple(muscle.endpoint2)))
            else:
                
                if muscle.wraps == True:
                    muscle_segs.append((tuple(muscle.endpoint1), 
                                        tuple(muscle.wrap_point), 
                                        tuple(muscle.endpoint2)))
                else:
                    muscle_segs.append((tuple(muscle.endpoint1), 
                                        tuple(muscle.endpoint2)))
                
        if muscle_tracker is not None:
            acts = []
            for muscle_data in muscle_tracker.muscles:
                acts.append(muscle_data.activation[0])
                
            colors = []
            for act in acts:
                red = act
                blue = 1-act
                green = 0
                alpha = 1
                colors.append((red, blue, green, alpha))
            
            muscle_lines = LineCollection(muscle_segs, linewidth=2, 
                                          colors=colors, zorder=1)
            
        else:
            muscle_lines = LineCollection(muscle_segs, linewidth=2, 
                                          color='r', zorder=1)
        
        # will hold all joints and hand to be plotted as circles
        circles = []
        
        # add circles representing each joint
        for joint in self.skeleton.joints:
            circle = patches.Circle((joint.location[0], joint.location[1]), 
                                    joint.diameter/2, fill=True)
            circles.append(circle)
            
        # add circle representing hand
        circle = patches.Circle((self.skeleton.bones[-1].endpoint2.coords[0], 
                                 self.skeleton.bones[-1].endpoint2.coords[1]), 
                                0.015, fill=True)
        circles.append(circle)
        
        circles_collection = PatchCollection(circles, color='k', zorder=3)
        
        # add visualization elements for first frame
        ax.add_collection(line_segments)
        ax.add_collection(muscle_lines)
        ax.add_collection(circles_collection)
        for marker in markers:
            ax.plot(marker[0][0], marker[0][1], marker='s', color='purple',
                    markersize=marker[1], zorder=4)
        
        # other stuff that needs to be passed to our animation function
        fargs = self, joint_angles, num_data_frames, num_video_frames, muscle_tracker
        
        # function that will be called for each frame of animation
        def func(frame, *fargs):
            
            # convert video frame number to data frame number
            data_frame = round((frame/num_video_frames)*num_data_frames)
            
            # set skeleton
            self.skeleton.write_joint_angles(joint_angles[data_frame])
            self.musculature.update_muscle_endpoints()
            
            # will hold all bones to be plotted as line segments
            segs = []
            
            markers = []
        
            # anchor for first bone
            anchor_x = self.skeleton.bones[0].endpoint1.coords[0]
            anchor_y = self.skeleton.bones[0].endpoint1.coords[1]
            segs.append(((anchor_x, anchor_y + 0.05), (anchor_x, anchor_y - 0.05)))
            
            # add all bones to the collection of segments to be plotted
            for bone in self.skeleton.bones:
                segs.append((tuple(bone.endpoint1.coords), tuple(bone.endpoint2.coords)))
                if bone.point_mass[0] != 0:
                    markers.append([bone.point_mass_loc, bone.point_mass[0]])
                
            muscle_segs = []
        
            # add all muscles to a collection of segments to be plotted
            for muscle in self.musculature.muscles:
                if isinstance(muscle, BiarticularMuscle):
                    muscle_segs.append((tuple(muscle.endpoint1), 
                                        tuple(muscle.wrap_point[0]), 
                                        tuple(muscle.wrap_point[1]),
                                        tuple(muscle.endpoint2)))
                else:
                    
                    if muscle.wraps == True:
                        muscle_segs.append((tuple(muscle.endpoint1), 
                                            tuple(muscle.wrap_point), 
                                            tuple(muscle.endpoint2)))
                    else:
                        muscle_segs.append((tuple(muscle.endpoint1), 
                                            tuple(muscle.endpoint2)))
                
            # will hold all joints and hand to be plotted as circles
            circles = []
            
            # add circles representing each joint
            for joint in self.skeleton.joints:
                circle = patches.Circle((joint.location[0], joint.location[1]), 
                                        joint.diameter/2, fill=True)
                circles.append(circle)
                
            # add circle representing hand
            circle = patches.Circle((self.skeleton.bones[-1].endpoint2.coords[0], 
                                     self.skeleton.bones[-1].endpoint2.coords[1]), 
                                    0.015, fill=True)
            circles.append(circle)
            
            if muscle_tracker is not None:
                acts = []
                for muscle_data in muscle_tracker.muscles:
                    acts.append(muscle_data.activation[data_frame])
                    
                colors = []
                for act in acts:
                    red = act
                    green = 0
                    blue = 1-act
                    alpha = 1
                    colors.append((red, green, blue, alpha))
                    
                muscle_lines.set_color(colors)
            
            # update and plot line and circle collections
            line_segments.set_paths(segs)
            muscle_lines.set_paths(muscle_segs)
            circles_collection.set_paths(circles)
            
            for line in ax.lines:
                line.set_marker(None)
            for marker in markers:
                ax.plot(marker[0][0], marker[0][1], marker='s', color='purple',
                    markersize=marker[1], zorder=4)
            
        # animate each frame using animation function
        anim = FuncAnimation(fig, func, frames=num_video_frames, 
                             interval=1/video_fps)
        
        # save the animation as a video
        if save == True:
            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=video_fps, metadata=dict(artist='Me'), bitrate=3000)
            anim.save(self.name + '.mp4', writer=writer)
        
        return anim
    
    # necessary to be able to pickle this object
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    
# experiment object, keeps track of data through various processing scripts
class Experiment:
    
    def __init__(self, name, f_s, associated_model):
        
        self.f_s = f_s
        self.T = 1/f_s
        self.name = name
        self.associated_model = associated_model
        self.t = []
        self.reach_times = []
        self.rest_time = []
        self.reach_duration = []
        self.endpoints = []
        self.joints = []
        
        # hard coded, fuck it
        self.forearm_CoM = []
        self.PG_vector = []
        self.elbow_loc = []
        self.elbow_loc_interp_x = None
        self.elbow_loc_interp_y = None
        self.inertial_torque = []
        self.inertial_torque_interp = None
        
    # returns torques needed at each joint to produce joint angles calculated 
    # by IK at a given time t, TAKING GRAVITY INTO ACCOUNT
    def return_torques(self, t):
        
        # set model to pose at time t and then update all necessary parameters
        joint_angles = []
        for joint in self.joints:
            joint_angles.append(joint.angle_interp(t))
        self.associated_model.skeleton.write_joint_angles(joint_angles)
        self.associated_model.skeleton.calc_I()
        tau_gravity = self.associated_model.skeleton.calc_gravity()
        
        # temp required torques list for each joint
        torques = []
        
        # inertial torque at this time
        i_torque = self.inertial_torque_interp(t)
        
        for i, joint in enumerate(self.joints):
            acceleration = joint.angle_interp.derivative(n=2)(t)
            torque = acceleration * self.associated_model.skeleton.joints[i].I
            torques.append(torque)
            
        final_torques = [a_i - b_i for a_i, b_i in zip(torques, tau_gravity)]
        final_torques[1] = final_torques[1] + i_torque
        return final_torques
    
    def return_torques_no_grav(self, t):
        
        torques = []
        
        for i, joint in enumerate(self.joints):
            acceleration = joint.angle_interp.derivative(n=2)(t)
            torque = acceleration * self.associated_model.skeleton.joints[i].I
            torques.append(torque)
            
        return torques
        
    # store self in working directory using pickle
    def dump(self):
        with open(self.name, "wb") as fp:
            pickle.dump(self, fp)
            
    # only works for ID right now
    def load_kinematic_data(self, kinematic_data):
        self.joints = []
        for i, joint_data in enumerate(kinematic_data.data):
            self.joints.append(kinematic_data.data[i])
            
        self.t = kinematic_data.t
            
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
            
        elif data == 'inertial_torque':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('tau (N*m)')
            ax.set_title(self.name + ' Inertial Torque (elbow)')
            
            labels = []
            
            plt.plot(self.t, self.inertial_torque)
               
            ax.legend(labels)
        
    # necessary to be able to pickle this object
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

class Simulation:

    def __init__(self, name, model, f_s):
        
        self.name = name
        self.model = model
        self.f_s = f_s
        self.T = 1/f_s
        self.joints = []
        self.t = [0]
        
        for joint in self.model.skeleton.joints:
            self.joints.append(JointData(joint.name))
            
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
    
# stores all muscle related data over the course of an experiment
class MuscleTracker:
    
    def __init__(self, name, muscles, experiment, associated_model):
        self.name = name
        self.muscles = muscles       
        self.experiment = experiment
        self.t = experiment.t
        self.associated_model = associated_model
        
    def calc_muscle_lengths(self):
        for i, t in enumerate(self.experiment.t):
            
            joint_angles = []
            for joint_data in self.experiment.joints:
                joint_angles.append(joint_data.angle[i])
            
            self.associated_model.skeleton.write_joint_angles(joint_angles)
            self.associated_model.musculature.update_muscle_endpoints()
            
            for muscle_data in self.muscles:
                for muscle in self.associated_model.musculature.muscles:
                    if muscle.name == muscle_data.name:
                        muscle_data.length.append(muscle.length)
                        
        for i, t in enumerate(self.experiment.t):
            
            joint_angles = []
            for joint_data in self.experiment.joints:
                joint_angles.append(joint_data.angle[i])
                
            self.associated_model.skeleton.write_joint_angles(joint_angles)
            self.associated_model.musculature.update_muscle_endpoints()
            
            # saves passive force generation over course of experiment in case
            # you want to look at it later
            for muscle_data in self.muscles:
                for muscle in self.associated_model.musculature.muscles:
                    
                    if muscle.name == muscle_data.name:
                        l_m = muscle.length/muscle.optimal_fiber_length
                        K_T = 0.5
                    
                        if l_m < 1:
                            MT_passive = 0
                        else:
                            MT_passive = K_T * ((l_m - 1)**2) * \
                                muscle.F_max * \
                                    muscle.moment_arm
                        muscle_data.MT_passive.append(MT_passive)
                        
        for muscle_data in self.muscles:
            muscle_data.length_interp = UnivariateSpline(self.t, muscle_data.length, s=0)
            
    def calc_muscle_moment_arms(self):
        
        for i, t in enumerate(self.experiment.t):
            
            joint_angles = []
            for joint_data in self.experiment.joints:
                joint_angles.append(joint_data.angle[i])
            
            self.associated_model.skeleton.write_joint_angles(joint_angles)
            self.associated_model.musculature.update_muscle_endpoints()
            self.associated_model.musculature.update_muscle_moment_arms()
            
            for muscle_data in self.muscles:
                for muscle in self.associated_model.musculature.muscles:
                    if muscle.name == muscle_data.name:
                        muscle_data.moment_arm.append(muscle.moment_arm)
                        
        for muscle_data in self.muscles:
            for muscle in self.associated_model.musculature.muscles:
                    if muscle.name == muscle_data.name:
                        muscle_data.moment_arm_interp = UnivariateSpline(self.t, muscle_data.moment_arm, s=0)
        
    # returns torques needed at each joint to produce joint angles calculated 
    # by IK at a given time t, TAKING GRAVITY INTO ACCOUNT, USING MUSCLES NOT
    # IDEALIZED TORQUES
    def return_torques(self, t, x_vec):
        
        # set the model to its current position in the forward simulation and 
        # adjust all relevant values
        joint_angles = x_vec
        self.associated_model.skeleton.write_joint_angles(joint_angles)
        self.associated_model.musculature.update_muscle_endpoints()
        self.associated_model.musculature.update_muscle_moment_arms()
        
        # list of lists containing each joint's name, diameter, rotation, and 
        # final applied torque value
        joint_torques = []
        for joint in self.associated_model.skeleton.joints:
            joint_torques.append([joint.name, joint.diameter, joint.rotation, 0])
            
        # nested for loop which matches muscles with their joints based on the 
        # joint name
        for muscle_data in self.muscles:
            if muscle_data.muscle.biarticular == True:
                for i, muscle_joint in enumerate(muscle_data.muscle.joint):
                    for joint_torque in joint_torques:
                        if muscle_joint == joint_torque[0]:
                            
                            # retrieves the muscle's activation at this time
                            muscle_act = muscle_data.forward_act_interp(t)
                            
                            # identify current muscle's maximum isometric force and 
                            # optimal fiber length (used to calculated normalized 
                            # fiber length)
                            muscle_Fmax = muscle_data.muscle.F_max
                            muscle_OFL = muscle_data.muscle.optimal_fiber_length
                            l_m = muscle_data.muscle.length/muscle_OFL
                            
                            # make sure this matches K_T in ID
                            K_T = 0.5
                            
                            # passive muscle force generation calculation
                            if l_m < 1:
                                MT_passive = 0
                            else:
                                MT_passive = K_T * ((l_m - 1)**2) * \
                                    muscle_Fmax * \
                                        muscle_data.muscle.moment_arm
                                        
                            # calculation of active muscle force-length factor
                            F_l = np.exp(((-1) * ((l_m - 1)**2))/0.45)
                                    
                            # check if joint rotation and muscle rotation directions 
                            # line up, then add in this muscle's torque contribution 
                            # accordingly
                            if muscle_data.muscle.rotation[i] == joint_torque[2]:
                                joint_torque[3] += (muscle_act * muscle_Fmax * \
                                    (muscle_data.muscle.moment_arm)* F_l) + MT_passive
                            else:
                                joint_torque[3] += (muscle_act * muscle_Fmax * \
                                    (muscle_data.muscle.moment_arm) * F_l * (-1)) - MT_passive
                                    
            else:
                for joint_torque in joint_torques:
                    if muscle_data.muscle.joint == joint_torque[0]:
                        
                        # retrieves the muscle'ss activation at this time
                        muscle_act = muscle_data.forward_act_interp(t)
                        
                        # identify current muscle's maximum isometric force and 
                        # optimal fiber length (used to calculated normalized 
                        # fiber length)
                        muscle_Fmax = muscle_data.muscle.F_max
                        muscle_OFL = muscle_data.muscle.optimal_fiber_length
                        l_m = muscle_data.muscle.length/muscle_OFL
                        
                        # make sure this matches K_T in ID
                        K_T = 0.5
                        
                        # passive muscle force generation calculation
                        if l_m < 1:
                            MT_passive = 0
                        else:
                            MT_passive = K_T * ((l_m - 1)**2) * \
                                muscle_Fmax * \
                                    muscle_data.muscle.moment_arm
                                    
                        # calculation of active muscle force-length factor
                        F_l = np.exp(((-1) * ((l_m - 1)**2))/0.45)
                                
                        # check if joint rotation and muscle rotation directions 
                        # line up, then add in this muscle's torque contribution 
                        # accordingly
                        if muscle_data.muscle.rotation == joint_torque[2]:
                            joint_torque[3] += (muscle_act * muscle_Fmax * \
                                (muscle_data.muscle.moment_arm)* F_l) + MT_passive
                        else:
                            joint_torque[3] += (muscle_act * muscle_Fmax * \
                                (muscle_data.muscle.moment_arm) * F_l * (-1)) - MT_passive

        torques = [el[3] for el in joint_torques]
        
        return torques 
    
    def trim_acts(self):
        for muscle_data in self.muscles:
            
            min_act = muscle_data.muscle.min_act
            max_act = muscle_data.muscle.max_act
            for i, act in enumerate(muscle_data.forward_act):
                if act < min_act:
                    muscle_data.forward_act[i] = min_act
                elif act > max_act:
                    muscle_data.forward_act[i] = max_act
               
    def plot(self, data):
        if data == 'length':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('l (m)')
            ax.set_title(self.name + ' Muscle Lengths')
            
            labels = []
            for muscle_data in self.muscles:
                plt.plot(self.t, muscle_data.length)
                labels.append(muscle_data.name)
                
            ax.legend(labels)
            
        elif data == 'activation':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('a')
            ax.set_ylim(0, 1)
            ax.set_title(self.name + ' Muscle Activations')
            
            labels = []
            for muscle_data in self.muscles:
                plt.plot(self.t, muscle_data.activation)
                labels.append(muscle_data.name)
                
            ax.legend(labels)
        
        elif data == 'excitation':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('u')
            ax.set_ylim(0, 1)
            ax.set_title(self.name + ' Muscle Excitations')
            
            labels = []
            for muscle_data in self.muscles:
                plt.plot(self.t, muscle_data.excitation)
                labels.append(muscle_data.name)
                
            ax.legend(labels)
            
        elif data == 'forward_act':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('a')
            ax.set_ylim(0, 1)
            ax.set_title(self.name + ' Forward Muscle Activations')
            
            labels = []
            for muscle_data in self.muscles:
                plt.plot(self.t, muscle_data.forward_act)
                labels.append(muscle_data.name)
                
            ax.legend(labels)
            
        elif data == 'passive_torque':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('tau (N*m)')
            ax.set_title(self.name + ' Passive Torque Generation')
            
            labels = []
            for muscle_data in self.muscles:
                plt.plot(self.t, muscle_data.MT_passive)
                labels.append(muscle_data.name)
                
            ax.legend(labels)
            
        elif data == 'moment_arms':
            
            # initialize figure
            fig, ax = plt.subplots()
            
            ax.set_xlabel('t (s)')
            ax.set_ylabel('l (m)')
            ax.set_title(self.name + ' Moment Arms')
            
            labels = []
            for muscle_data in self.muscles:
                plt.plot(self.t, muscle_data.moment_arm)
                labels.append(muscle_data.name)
                
            ax.legend(labels)
            
    # store self in working directory using pickle
    def dump(self):
        with open(self.name, "wb") as fp:
            pickle.dump(self, fp)
              
    # necessary to be able to pickle this object
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    
class MuscleData:
    def __init__(self, name, muscle):
        
        self.name = name
        self.muscle = muscle
        
        self.length = []
        self.length_interp = None
        
        self.activation = []
        self.activation_interp = None
        
        self.excitation = []
        self.forward_excite = []
        self.forward_excite_interp = None
        
        self.forward_act = []
        self.forward_act_interp = None
        
        self.MT_passive = []
        
        self.moment_arm = []
        self.moment_arm_interp = None

# tracking data related to a single joint through an experiment
class JointData:
    
    def __init__(self, name):
    
        self.name = name
        
        self.angle = []
        self.velocity = [0]
        self.acceleration = []
        
        self.torque = []
        self.torque_no_grav = []
        
        self.angle_interp = None
        
class ExcitationData:
    
    def __init__(self, muscle_name, u):
        self.muscle_name = muscle_name
        self.u = u
        
class KinematicData:
    
    def __init__(self, name, experiment):
        
        self.name = name
        self.data = []
        self.t = experiment.t
        
        for joint_data in experiment.joints:
            self.data.append(joint_data)
            
    # store self in working directory using pickle
    def dump(self):
        with open(self.name, "wb") as fp:
            pickle.dump(self, fp)
              
    # necessary to be able to pickle this object
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
            
        
    
        

        

