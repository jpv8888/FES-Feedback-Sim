# -*- coding: utf-8 -*-
"""
Generates artifical endpoint movement data, i.e. the location of the hand 
through time
@author: Jack Vincent
"""

import itertools
import pickle
from math import dist, radians
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import logistic

import alphashape
from descartes import PolygonPatch
from shapely.geometry import Point, LineString
from sklearn.linear_model import LinearRegression

import model

# generate fresh skeleton
skeleton = model.init()

# %% explore endpoint solution space 

# degrees, precision with which to comb through possible endpoint locations
angle_step = 1

# will eventually hold all possible joint angle combinations
angle_space = []

# outputs list of lists, where each list corresponds to all possible values of
# a joint angle
for joint in skeleton.joints:
    angle_space.append(np.arange(radians(joint.min_ang), 
                                 radians(joint.max_ang) + radians(angle_step), 
                                 radians(angle_step)))
    
# converts angle_space list to an iterable of every possible combination of 
# joint angles
angle_space = itertools.product(*angle_space)
   
# records every hand endpoint associated with the previously generated angles
possible_endpoints = []
for joint_angles in angle_space:
    skeleton.write_joint_angles(joint_angles)
    possible_endpoints.append(skeleton.bones[-1].endpoint2.coords)
    
# %% generate alpha shape associated with endpoint solution space
  
# optimal alpha can be calculated automatically, but doing this results in a 
# very long run time; user-selected alpha should be big enough to tightly wrap
# around the solution space but low enough to not exclude points
alpha = 7

# generate and plot solution space alpha shape
alpha_shape = alphashape.alphashape(possible_endpoints, alpha)

# axis limits
left = min(endpoint[0] for endpoint in possible_endpoints)
right = max(endpoint[0] for endpoint in possible_endpoints)
top = max(endpoint[1] for endpoint in possible_endpoints)
bottom = min(endpoint[1] for endpoint in possible_endpoints)

# figure formatting
fig, ax = plt.subplots()
ax.axis('square')
ax.set_xlim(left - 0.1, right + 0.1)
ax.set_ylim(bottom - 0.1, top + 0.1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Possible Hand Endpoint Locations')

# add the possible hand endpoints
ax.scatter(*zip(*possible_endpoints), 0.1)

# add a circle to show where the shoulder is
circle = patches.Circle((skeleton.joints[0].location[0], skeleton.joints[0].location[1]), 
                        0.015, fill=True)
ax.add_patch(circle)
plt.annotate('shoulder', 
             (skeleton.joints[0].location[0], skeleton.joints[0].location[1]))

# add the alpha shape
ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))  

# %% generate random endpoints contained in the solution space

# Morasso P. Spatial control of arm movements. Exp Brain Res. 
# 1981;42(2):223-227. doi:10.1007/BF00236911
# reaching position profiles generally follow straight lines from target to 
# target, at least in the tranverse plane
# this model is constructed in the sagittal plane but we still assume straight 
# lines between reaching targets

# space we will use to randomly try points
xlims = ax.get_xlim()
ylims = ax.get_ylim()

# desired total number of endpoints to generate
num_endpoints = 5

# will hold final endpoint selections
endpoints = []

# this loop just finds a first endpoint to start with
while len(endpoints) < 1:
    
    # generate possible endpoint
    x = np.random.uniform(xlims[0], xlims[1])
    y = np.random.uniform(ylims[0], ylims[1])
    point = Point(x, y)
    
    # as long as it's in the alpha shape, we're good to go
    if alpha_shape.contains(point):
        endpoints.append((x, y))
        previous_endpoint = (x, y)
        
# this loop finds all subsequent endpoints past the first one
while len(endpoints) < num_endpoints:
    
    # generate possible endpoint
    x = np.random.uniform(xlims[0], xlims[1])
    y = np.random.uniform(ylims[0], ylims[1])
    point = Point(x, y)
    
    # if it's contained in the alpha shape, move to the next step
    if alpha_shape.contains(point):
        current_endpoint = (x, y)
        
        # generate a line connecting previous endpoint to this potential 
        # endpoint, if it doesn't intersect the boundary of the alpha shape 
        # (i.e. is fully contained within the alpha shape), we're good to go
        movement_line = LineString([previous_endpoint, current_endpoint])
        if not movement_line.intersects(alpha_shape.boundary):
            endpoints.append(current_endpoint)
            previous_endpoint = current_endpoint
            
# figure formatting
fig, ax = plt.subplots()
ax.axis('square')
ax.set_xlim(left - 0.1, right + 0.1)
ax.set_ylim(bottom - 0.1, top + 0.1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Random Hand Endpoint Reaching Targets')

# add a circle to show where the shoulder is
circle = patches.Circle((skeleton.joints[0].location[0], skeleton.joints[0].location[1]), 
                        0.015, fill=True)
ax.add_patch(circle)
plt.annotate('shoulder', 
             (skeleton.joints[0].location[0], skeleton.joints[0].location[1]))

# add the alpha shape
ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2)) 

# add a circle to highlight the location of the first and last endpoint
circle = patches.Circle((endpoints[0][0], endpoints[0][1]), 0.01, fill=True)
ax.add_patch(circle)
circle = patches.Circle((endpoints[-1][0], endpoints[-1][1]), 0.01, fill=True)
ax.add_patch(circle)

# plot the generated endpoints with arrows, so it's clear what direction the 
# hand will be moving in
i = 1
while i < len(endpoints):
    plt.arrow(endpoints[i - 1][0], endpoints[i - 1][1], 
              endpoints[i][0] - endpoints[i - 1][0], 
              endpoints[i][1] - endpoints[i - 1][1], head_width=0.02, 
              length_includes_head=True)
    i += 1

# %% linear regression: relationship between reach distance and peak velocity

""" 
initially I had planned to use normal distributions with peaks calculated from
this linear regression and bounds dictated by reach time (the famous 
bell-shaped velocity curves) to generate cumulative distribution functions 
that would describe the motion of hand from point to point, but that proved 
more complicated than anticipated, so that motion is currently described by a 
generic sigmoid
this information may still be used at some point in the future
""" 

# Morasso P. Spatial control of arm movements. Exp Brain Res. 
# 1981;42(2):223-227. doi:10.1007/BF00236911
# hand velocity profiles are bell shaped during reaching tasks with a peak 
# velocity roughly proportional to the total distance of the reach
# per this source, for reaches of distance: 35, 41.7, and 62.2 cm; associated
# peak hand velocities: 65, 78, and 88 cm/s 

"""
these values might not be correct; also there's a 4th data point that could be
extracted from the paper to make the regression more accurate
"""

# data importing and formatting
X = np.array([0.35, 0.417, 0.622])
X = X.reshape(-1, 1)
y = np.array([0.65, 0.78, 0.88])
y = y.reshape(-1, 1)

# run linear regression since peak velocity is supposedly proportional to 
# reach distance; no intercept since obviously peak velocity should be 0 for a
# reach distance of 0
reg = LinearRegression(fit_intercept=False).fit(X, y)

# will hold reach distances for each reach between generated endpoints
reach_distances = []

# calculate reach distances
i = 1
while i < len(endpoints):
    reach_distances.append(dist(endpoints[i], endpoints[i - 1]))
    i += 1
    
# calculate associated peak velocities according to linear regression
peak_velocities = reg.predict(np.array(reach_distances).reshape(-1, 1))

# %% generate discrete position data over time

# Morasso P. Spatial control of arm movements. Exp Brain Res. 
# 1981;42(2):223-227. doi:10.1007/BF00236911
# each reach took approximately the same time (1 s), despite differences in 
# reach distance (enabled by varying peak velocity), so we will treat total 
# reach time as a constant

# time each reach will take
reach_time = 1.5

# maximum rest time between reaches
max_rest_time = 1

# sampling frequency and period
f_s = 100
T = 1/f_s

# time vector for reach
t_reach = np.arange(0, reach_time + T, T)

# will hold values outlining generic sigmoid 
sig = []

# populate sigmoid curve
for time in t_reach:
    sig.append(logistic.cdf(time, loc=reach_time/2, scale=0.1))

"""
scale stretches the sigmoid out if it is increased; in order to fully automate
this process, it should automatically be calculated based on the reach time, 
but how exactly this should be done has not yet been determined
"""

# these values must be hard set because sigmoid curve asymptotes, however 
# first value of any reach should be whatever the endpoint was previously
# while at rest, and last value of any reach should be the new rest value
sig[0] = 0
sig[-1] = 1

# lists that will hold final time and endpoint values with their initial 
# values already inserted 
t = []
t.append(0)
endpoints_final = []
endpoints_final.append(endpoints[0])

# runs for as many iterations as there are endpoints - 1, i.e. doesn't run for
# the last one because there is no subsequent reach
for i in range(len(endpoints) - 1):
    
    # generate and populate rest period of random length, round t values or 
    # error will start to accumulate 
    rest_time = np.random.uniform(0, max_rest_time)
    last_end_time = t[-1]
    while t[-1] < last_end_time + rest_time:
        t.append(round(t[-1] + T, 3))
        endpoints_final.append(endpoints_final[-1])
        
    # the change that will take place in the x and y coordinates over the 
    # course of the reach
    x_transform = endpoints[i + 1][0] - endpoints[i][0]
    y_transform = endpoints[i + 1][1] - endpoints[i][1]
    
    # starting x and y values pre-reach
    old_x = endpoints_final[-1][0]
    old_y = endpoints_final[-1][1]
    
    # connect pre and post reach endpoints with a sigmoidal movement curve
    for scale in sig:
        t.append(round(t[-1] + T, 3))
        endpoints_final.append((scale*x_transform + old_x, scale*y_transform + old_y))
    
# append one last random rest period
rest_time = np.random.uniform(0, max_rest_time)
last_end_time = t[-1]
while t[-1] < last_end_time + rest_time:
    t.append(round(t[-1] + T, 3))
    endpoints_final.append(endpoints_final[-1])

# write generated time and hand endpoint position data
with open("t.txt", "wb") as fp:
    pickle.dump(t, fp)
with open("endpoints.txt", "wb") as fp:
    pickle.dump(endpoints_final, fp)
    
"""
ultimately, a more sophisticated data structure is going to be needed to keep 
track of all the data recorded over the course of an experiment (e.g. joint 
angles, velocities, muscle activations, etc.), including this initial data;  
until then, we're just writing it out of this script and reading it into 
whatever script it might be needed in with pickle
"""
