# -*- coding: utf-8 -*-
"""
Inverse Kinematics: calculates joint angles needed to produce input movement 
of hand endpoint through time via multivariate function minimization 
@author: Jack Vincent, Yassin Fahmy 
"""

import math
from scipy.optimize import Bounds, minimize
from progressbar import ProgressBar, Percentage, Bar, ETA

import model

# %% constants

# relative penalization factors for sources of error
endpoint_pen = 100
ang_dev_pen = 2

# %% data preparation prior to generating poses

# load in model and experiment
current_model = model.load('upper_arm_0')
current_experiment = model.load('8-17-20')
endpoints = current_experiment.endpoints

# this list will hold the final joint angles calculated to produce the desired 
# hand endpoint locations
IK_joint_angles = []

# bounds on joint angles (dictated by anatomy), used by both initial_pose() 
# and more_pose() (lb is lower bound and ub is upper bound)
lb = []
ub = []
for joint in current_model.skeleton.joints:
    lb.append(math.radians(joint.min_ang))
    ub.append(math.radians(joint.max_ang))

# Bounds function in scipy    
bounds = Bounds(lb, ub)

# %% calculate the initial pose of the skeleton

# initial guess is midpoint of joint range of motion
x0 = []
for joint in current_model.skeleton.joints:
    x0.append((math.radians(joint.max_ang) - math.radians(joint.min_ang))/2) 

# desired hand endpoint to solve for (the starting endpoint)
hand_endpoint = endpoints[0]

# stuff the function we're minimizing needs to have access to
args = (current_model.skeleton, hand_endpoint, endpoint_pen, ang_dev_pen)

# function to be minimized 
def initial_pose(joint_angles, *args):

    # write input joint angles and determine subsequent hand endpoint location
    current_model.skeleton.write_joint_angles(joint_angles)
    model_endpoint = current_model.skeleton.bones[-1].endpoint2.coords

    # 2 sources of error: distance of calculated model endpoint from desired
    # endpoint and deviation of joint angles from their default values
    endpoint_error = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(model_endpoint, hand_endpoint)))
    default_ang_dev = 0
    for i, joint_angle in enumerate(joint_angles):
        default_ang_dev += abs(joint_angle - math.radians(current_model.skeleton.joints[i].default))

    # endpoint error is weighted much more heavily, as minimizing deviation 
    # from default joint angles is secondary to getting the endpoint correct
    return endpoint_pen*endpoint_error + ang_dev_pen*default_ang_dev

# minimize function using Sequential Least Squares Programming (SLSQP)
res = minimize(initial_pose, x0, method='SLSQP', options={'verbose': 0, 'ftol': 1/10**10},
               bounds=bounds)

# write solution to skeleton
for i, joint in enumerate(current_model.skeleton.joints):
    joint.angle = res.x[i]

# record solution
IK_joint_angles.append(current_model.skeleton.return_joint_angles())

# %% calculate all other poses

# function to be minimized 
def more_pose(joint_angles, *args):

    # write input joint angles and determine subsequent hand endpoint location
    current_model.skeleton.write_joint_angles(joint_angles)
    model_endpoint = current_model.skeleton.bones[-1].endpoint2.coords

    # 1 source of error: distance of calculated model endpoint from desired
    # endpoint
    endpoint_error = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(model_endpoint, hand_endpoint)))

    # endpoint error, only source of error: initial guess is previous pose of 
    # skeleton, so solver should automatically find solution with least angle
    # deviation 
    return endpoint_pen*endpoint_error

# skip the first endpoint since we've already calculated the initial pose
iter_endpoints = iter(endpoints)
next(iter_endpoints)

# widgets for the progress bar
widgets = ['PROGRESS: ', Percentage(), ' ',
              Bar(marker='-',left='[',right=']\n'),
               ' ', ETA(),' \n ']

# create a progress bar object
pbar = ProgressBar(maxval=len(endpoints),widgets=widgets).start()

# iterate through every endpoint (other than first one)
for i, endpoint in enumerate(iter_endpoints):
    
    # update progress bar
    pbar.update(i+1)

    # initial guess is current value of all joint angles
    x0 = []
    for joint in current_model.skeleton.joints:
        x0.append(joint.angle)

    # current endpoint being solved for
    hand_endpoint = endpoint

    # stuff the function we're minimizing needs to have access to
    args = (current_model.skeleton, hand_endpoint, x0, endpoint_pen)

    # minimize function using trust regions subject to constraints
    res = minimize(more_pose, x0, method='SLSQP', 
                   options={'verbose': 0, 'ftol': 1/10**10}, bounds=bounds)

    # write solution to skeleton
    for i, joint in enumerate(current_model.skeleton.joints):
        joint.angle = res.x[i]

    # record solution
    IK_joint_angles.append(current_model.skeleton.return_joint_angles())
    
# end progress bar    
pbar.finish() 

# create joint data objects for experiment
joint_datas = []
for joint in current_model.skeleton.joints:
    joint_datas.append(model.JointData(joint.name))

current_experiment.joints = joint_datas

# write IK data to experiment
for i, joint_data in enumerate(current_experiment.joints):
    
    # isolate IK angle data for 1 joint
    joint_angles = [x[i] for x in IK_joint_angles]
        
    # write joint angle data to experiment
    joint_data.angle = joint_angles

# examine the finished results
anim = current_model.animate(current_experiment)

# write finished results
current_model.dump()
current_experiment.dump()