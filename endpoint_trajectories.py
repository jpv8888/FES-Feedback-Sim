# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:49:53 2020

@author: 17049
"""

import alphashape
from descartes import PolygonPatch
import itertools
from math import radians
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point

import model

skeleton = model.init()

# %% explore endpoint solution space 

# degrees
angle_step = 1

angle_space = []

for joint in skeleton.joints:
    angle_space.append(np.arange(radians(joint.min_ang), 
                                 radians(joint.max_ang) + radians(angle_step), 
                                 radians(angle_step)))
    
angle_space = itertools.product(*angle_space)
   
possible_endpoints = []
for joint_angles in angle_space:
    skeleton.write_joint_angles(joint_angles)
    possible_endpoints.append(skeleton.bones[-1].endpoint2.coords)
    
# %% generate alpha shape associated with endpoint solution space
  
# optimal alpha can be calculated automatically, but doing this results in a 
# very long run time, user-selected alpha should be big enough to tightly wrap
# around the solution space but low enough to not exclude points
alpha = 7

alpha_shape = alphashape.alphashape(possible_endpoints, alpha)
fig, ax = plt.subplots()
ax.axis("equal")
ax.scatter(*zip(*possible_endpoints), 0.1)
circle = patches.Circle((skeleton.joints[0].location[0], skeleton.joints[0].location[1]), 
                        0.015, fill=True)
ax.add_patch(circle)
ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2)) 
     
plt.show()

# %% generate random endpoint pairs contained in the solution space 

xlims = ax.get_xlim()
ylims = ax.get_ylim()

num_pairs = 2

endpoint_pairs = []

while len(endpoint_pairs) < num_pairs:
    
    x = np.random.uniform(xlims[0], xlims[1])
    y = np.random.uniform(ylims[0], ylims[1])
    point = Point(x, y)
    if alpha_shape.contains(point):
        point1 = (x, y)
        x = np.random.uniform(xlims[0], xlims[1])
        y = np.random.uniform(ylims[0], ylims[1])
        point = Point(x, y)
        if alpha_shape.contains(point):
            point2 = (x, y)
            endpoint_pairs.append((point1, point2))            

