# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 08:30:01 2020

@author: 17049
"""

import numpy as np
from scipy.signal import butter, filtfilt

import model

current_model = model.load('upper_arm_0')
current_experiment = model.load('8-17-20')
current_muscle_tracker = model.load('test')

kinematic_data_name = 'triceps_lock'

mode = 'constant'
constant_joint_angles = [1.22, 3.14]

duration = 2

if mode == 'constant':
    # create joint data objects for experiment
    joint_datas = []
    for joint in current_model.skeleton.joints:
        joint_datas.append(model.JointData(joint.name))
    
    current_experiment.joints = joint_datas
    
    # Filter requirements.
    fs = current_experiment.f_s      # sample rate, Hz
    cutoff = 6      # desired cutoff frequency of the filter, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 2       # sin wave can be approx represented as quadratic
    n = len(current_experiment.t) # total number of samples
    
    def butter_lowpass_filter(data, cutoff, fs, order):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y
    
    t = np.arange(0, duration, 1/fs)
    current_experiment.t = t
    
    angles_tuple = tuple(constant_joint_angles)
    
    IK_joint_angles = [angles_tuple] * len(t) 
    
    # write IK data to experiment
    for i, joint_data in enumerate(current_experiment.joints):
        
        # isolate IK angle data for 1 joint
        joint_angles = [x[i] for x in IK_joint_angles]
        
        # fix bug caused by switching between initial_pose() and more_pose()
        if joint_angles[1] != joint_angles[0]:
            joint_angles[0] = joint_angles[1]
        
        y = butter_lowpass_filter(joint_angles, cutoff, fs, order)
            
        # write joint angle data to experiment
        joint_data.angle = y
            
    # examine the finished results
    anim = current_model.animate(current_experiment)
    
    kinematic_data = model.KinematicData(kinematic_data_name, current_experiment)
    kinematic_data.dump()
    
    # write finished results
    current_model.dump()
    current_experiment.dump()