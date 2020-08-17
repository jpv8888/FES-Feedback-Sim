# -*- coding: utf-8 -*-
"""
Inverse Kinematics: calculates joint angles needed to produce input movement 
of hand endpoint through time via multivariate function minimization 
@author: Jack Vincent
"""

import math
from scipy.optimize import Bounds, minimize

import model


# %% data preparation prior to generating poses

# load in model and experiment
current_model = model.load('upper_arm_0')
current_experiment = model.load('8-17-20')
endpoints = current_experiment.endpoints

# this list will hold the final joint angles calculated to produce the desired 
# hand endpoint locations
IK_joint_angles = []


# bounds on joint angles (dictated by anatomy), used by both initial_pose() 
# and more_pose() 
lb = []
ub = []
for joint in current_model.skeleton.joints:
    lb.append(math.radians(joint.min_ang))
    ub.append(math.radians(joint.max_ang))
    
bounds = Bounds(lb, ub)

# %% calculate the initial pose of the skeleton

# initial guess is all joint angles set to their default values
x0 = []
for joint in current_model.skeleton.joints:
    x0.append(math.radians(joint.default))

# desired hand endpoint to solve for (the starting endpoint)
hand_endpoint = endpoints[0]

# stuff the function we're minimizing needs to have access to
args = (current_model.skeleton, hand_endpoint)

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
    return 10*endpoint_error + default_ang_dev

# minimize function using trust regions subject to constraints
res = minimize(initial_pose, x0, method='trust-constr', options={'verbose': 0},
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
    
    # 2 sources of error: distance of calculated model endpoint from desired
    # endpoint and deviation of joint angles from their PREVIOUS values
    endpoint_error = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(model_endpoint, hand_endpoint)))
    ang_dev = sum(abs(px - qx) for px, qx in zip(joint_angles, x0))
    
    # endpoint error is weighted much more heavily, as minimizing deviation 
    # from previous joint angles is secondary to getting the endpoint correct
    return 10*endpoint_error + ang_dev

# skip the first endpoint since we've already calculated the initial pose
iter_endpoints = iter(endpoints)
next(iter_endpoints)

# iterate through every endpoint (other than first one)
for i, endpoint in enumerate(iter_endpoints):
    
    print(i)
    
    """
    this is a crude, temp way to track the IK solver's progress, a cleaner way
    would be nice
    """
    
    # initial guess is current value of all joint angles
    x0 = []
    for joint in current_model.skeleton.joints:
        x0.append(math.radians(joint.angle))

    # current endpoint being solved for
    hand_endpoint = endpoint
    
    # stuff the function we're minimizing needs to have access to
    args = (current_model.skeleton, hand_endpoint, x0)
    
    # minimize function using trust regions subject to constraints
    res = minimize(more_pose, x0, method='trust-constr', 
                   options={'verbose': 0}, bounds=bounds)
    
    """
    constantly throws this warning:
    UserWarning: delta_grad == 0.0. Check if the approximated function is 
    linear. If the function is linear better results can be obtained by 
    defining the Hessian as zero instead of using quasi-Newton approximations.
    need to find a way to silence this, or fix whatever it's complaining about
    initial_pose() has the same problem, but it's less conspicuous since it 
    only runs once
    """

    # write solution to skeleton
    for i, joint in enumerate(current_model.skeleton.joints):
        joint.angle = res.x[i]

    # record solution
    IK_joint_angles.append(current_model.skeleton.return_joint_angles())
    
# examine the finished results
anim = current_model.skeleton.animate(IK_joint_angles, 100)

for joint in current_model.skeleton.joints:
    current_experiment.joints.append(model.JointData(joint.name))

for i, joint_data in enumerate(current_experiment.joints):
    current_experiment.joints[i].angle = [x[i] for x in IK_joint_angles]

# write finished results
current_model.dump()
current_experiment.dump()
    

