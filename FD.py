# -*- coding: utf-8 -*-
"""
Forward Dynamics:

@author: Jack Vincent
"""

from math import radians
import numpy as np

import model

sim_name = 'test'
f_s = 100
duration = 2

# load in model and experiment
current_model = model.load('upper_arm_0')
current_experiment = model.load('8-17-20')

init_joint_angles = []
for joint_data in current_experiment.joints:
    init_joint_angles.append(joint_data.angle[0])
    
current_model.skeleton.write_joint_angles(init_joint_angles)

current_simulation = model.Simulation(sim_name, current_model, f_s, duration)

# first time step (t = 0)
# gravity_torques = current_model.skeleton.calc_gravity()
# for j, joint_data in enumerate(current_experiment.joints):
#     gravity_torques[j] += joint_data.torque[i]
# for i, joint_data in enumerate(current_simulation.joints):
#     joint_data.torque.append(gravity_torques[i])
# current_model.skeleton.calc_I()
# I = []
# for joint in current_model.skeleton.joints:
#     I.append(joint.I)
# a = list(np.divide(gravity_torques, I))
# for i, joint_data in enumerate(current_simulation.joints):
#     joint_data.acceleration.append(a[i])

# second time step (t = 0 + T)
# gravity_torques = current_model.skeleton.calc_gravity()
# for i, joint_data in enumerate(current_simulation.joints):
#     joint_data.torque.append(gravity_torques[i])
# current_model.skeleton.calc_I()
# I = []
# for joint in current_model.skeleton.joints:
#     I.append(joint.I)
# a = list(np.divide(gravity_torques, I))
# for i, joint_data in enumerate(current_simulation.joints):
#     joint_data.acceleration.append(a[i])
# for joint_data in current_simulation.joints:
#     joint_data.velocity.append(joint_data.velocity[0] + joint_data.acceleration[0]*current_simulation.T)


time_steps = int(duration/current_simulation.T)
itertime = iter(range(time_steps))
next(itertime)
next(itertime)

# for i in itertime:
#     gravity_torques = current_model.skeleton.calc_gravity()
#     joint_torques = gravity_torques
    
#     joint_torques = [0, 0]
    
    
#     for j, joint_data in enumerate(current_experiment.joints):
#         if i < len(joint_data.torque):
#             joint_torques[j] += joint_data.torque[i]
    
#     for j, joint_data in enumerate(current_simulation.joints):
#         joint_data.torque.append(joint_torques[j])
        
#     current_model.skeleton.calc_I()
#     I = []
#     for joint in current_model.skeleton.joints:
#         I.append(joint.I)
#     a = list(np.divide(joint_torques, I))
    
#     for j, joint_data in enumerate(current_simulation.joints):
#         joint_data.acceleration.append(a[j])
        
#     for joint_data in current_simulation.joints:
#         joint_data.velocity.append(joint_data.velocity[-1] + (joint_data.acceleration[-2]*current_simulation.T))
        
#     for joint_data in current_simulation.joints:
#         joint_data.angle.append(joint_data.angle[-1] + (joint_data.velocity[-2]*current_simulation.T))
        
#     angles = []
#     for joint_data in current_simulation.joints:
#         angles.append(joint_data.angle[-1])
#     current_model.skeleton.write_joint_angles(angles)
        
# angles = []
# for j, joint_data in enumerate(current_simulation.joints):
#     if joint_data.angle[-1] < radians(current_model.skeleton.joints[j].min_ang):
#         joint_data.angle[-1] = radians(current_model.skeleton.joints[j].min_ang)
#         joint_data.velocity[-1] = 0
#     elif joint_data.angle[-1] > radians(current_model.skeleton.joints[j].max_ang):
#         joint_data.angle[-1] = radians(current_model.skeleton.joints[j].max_ang)
#         joint_data.velocity[-1] = 0
#     angles.append(joint_data.angle[-1])

for i in itertime:
    
    angles = []
    for joint_data in current_simulation.joints:
        angles.append(joint_data.angle[-1] + (joint_data.velocity[-1]*current_simulation.T))
        joint_data.angle.append(joint_data.angle[-1] + (joint_data.velocity[-1]*current_simulation.T))
    current_model.skeleton.write_joint_angles(angles)
    
    for joint_data in current_simulation.joints:
        joint_data.velocity.append(joint_data.velocity[-1] + (joint_data.acceleration[-1]*current_simulation.T))
        
    gravity_torques = current_model.skeleton.calc_gravity()
    joint_torques = gravity_torques
    
    for j, joint_data in enumerate(current_experiment.joints):
        if i < len(joint_data.torque):
            joint_torques[j] += joint_data.torque[i]
    
    for j, joint_data in enumerate(current_simulation.joints):
        joint_data.torque.append(joint_torques[j])
        
    current_model.skeleton.calc_I()
    I = []
    for joint in current_model.skeleton.joints:
        I.append(joint.I)
    a = list(np.divide(joint_torques, I))
    
    for j, joint_data in enumerate(current_simulation.joints):
        joint_data.acceleration.append(a[j])
        
        
    
    
    



    
    