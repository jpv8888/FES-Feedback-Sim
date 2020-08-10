# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 23:18:32 2020

@author: 17049
"""

import model
from bones import *
from scipy.optimize import Bounds, minimize

wrist_endpoint = [-0.02, 0.2]

skeleton = model.init()

x0 = []
for joint in skeleton.joints:
    x0.append(math.radians(joint.default))
    
lb = []
ub = []
for joint in skeleton.joints:
    lb.append(math.radians(joint.min_ang))
    ub.append(math.radians(joint.max_ang))
    
bounds = Bounds(lb, ub)

args = (skeleton, wrist_endpoint)

def initial_pose(joint_angles, *args):
    
    skeleton.write_joint_angles(joint_angles)
    model_endpoint = skeleton.bones[-1].endpoint2.coords
    endpoint_error = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip(model_endpoint, wrist_endpoint)))
    default_ang_dev = 0
    for i, joint_angle in enumerate(joint_angles):
        default_ang_dev += abs(joint_angle - math.radians(skeleton.joints[i].default))
    
    return 10*endpoint_error + default_ang_dev

res = minimize(initial_pose, x0, method='trust-constr', options={'verbose': 1}, 
               bounds=bounds)

for i, joint in enumerate(skeleton.joints):
    joint.angle = res.x[i]
    
skeleton.visualize()
