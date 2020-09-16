# -*- coding: utf-8 -*-
"""
Generates artifical endpoint movement data, i.e. the location of the hand 
through time
@author: Jack Vincent
"""

from math import dist
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb

from descartes import PolygonPatch
from shapely.geometry import Point, LineString
from sklearn.linear_model import LinearRegression

import model

# load in model and experiment
current_model = model.load('upper_arm_0')
current_experiment = model.load('8-17-20')

# %% generate random endpoints contained in the solution space

# Morasso P. Spatial control of arm movements. Exp Brain Res. 
# 1981;42(2):223-227. doi:10.1007/BF00236911
# reaching position profiles generally follow straight lines from target to 
# target, at least in the tranverse plane
# this model is constructed in the sagittal plane but we still assume straight 
# lines between reaching targets

# desired total number of endpoints to generate
num_endpoints = 3

# space we will use to randomly try points
xlims = current_model.skeleton.x_lim
ylims = current_model.skeleton.y_lim

# will hold final endpoint selections
endpoints = []

# this loop just finds a first endpoint to start with
while len(endpoints) < 1:
    
    # generate possible endpoint
    x = np.random.uniform(xlims[0], xlims[1])
    y = np.random.uniform(ylims[0], ylims[1])
    point = Point(x, y)
    
    # as long as it's in the alpha shape, we're good to go
    if current_model.skeleton.alpha_shape.contains(point):
        endpoints.append((x, y))
        previous_endpoint = (x, y)
        
# this loop finds all subsequent endpoints past the first one
while len(endpoints) < num_endpoints:
    
    # generate possible endpoint
    x = np.random.uniform(xlims[0], xlims[1])
    y = np.random.uniform(ylims[0], ylims[1])
    point = Point(x, y)
    
    # if it's contained in the alpha shape, move to the next step
    if current_model.skeleton.alpha_shape.contains(point):
        current_endpoint = (x, y)
        
        # generate a line connecting previous endpoint to this potential 
        # endpoint, if it doesn't intersect the boundary of the alpha shape 
        # (i.e. is fully contained within the alpha shape), we're good to go
        movement_line = LineString([previous_endpoint, current_endpoint])
        if not movement_line.intersects(current_model.skeleton.alpha_shape.boundary):
            endpoints.append(current_endpoint)
            previous_endpoint = current_endpoint
            
# figure formatting
fig, ax = plt.subplots()
ax.axis('square')
ax.set_xlim(current_model.skeleton.x_lim[0], current_model.skeleton.x_lim[1])
ax.set_ylim(current_model.skeleton.y_lim[0], current_model.skeleton.y_lim[1])
ax.set_xlabel('x position')
ax.set_ylabel('y position')
ax.set_title('Random Hand Endpoint Reaching Targets \n (In sagittal section view)')

# add a circle to show where the shoulder is
circle = patches.Circle((current_model.skeleton.joints[0].location[0], current_model.skeleton.joints[0].location[1]), 
                        0.015, fill=True)
ax.add_patch(circle)
plt.annotate('Shoulder Joint Axis', 
             (current_model.skeleton.joints[0].location[0], current_model.skeleton.joints[0].location[1]))

# add the alpha shape
ax.add_patch(PolygonPatch(current_model.skeleton.alpha_shape, alpha=0.2))

# add a green circle to highlight the location of the first endpoint and name 
# it Starting position
circle = patches.Circle((endpoints[0][0], endpoints[0][1]), 0.015, fill=True)
circle.set_facecolor('lime')
plt.annotate('Starting position',(endpoints[0][0], endpoints[0][1])) 
ax.add_patch(circle)

# add a red circle to highlight the location of the last endpoint and name it 
# End position
circle = patches.Circle((endpoints[-1][0], endpoints[-1][1]), 0.015, fill=True)
circle.set_facecolor('red')
plt.annotate('End position',(endpoints[-1][0], endpoints[-1][1]))
ax.add_patch(circle)

# plot the generated endpoints with arrows, so it's clear what direction the 
# hand will be moving in
i = 1
while i < len(endpoints):
    plt.arrow(endpoints[i - 1][0], endpoints[i - 1][1], 
              endpoints[i][0] - endpoints[i - 1][0], 
              endpoints[i][1] - endpoints[i - 1][1], head_width=0.03, 
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
reach_time = 1

# rest time between reaches
rest_time = 1

# sampling frequency and period
f_s = current_experiment.f_s
T = current_experiment.T

# pythonic smoothstep function, shamelessly copied from Stack Overflow
def smoothstep(x, x_min=0, x_max=reach_time, N=5):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result

# time vector for reach
t_reach = np.arange(0, reach_time, T)

# will hold values outlining generic sigmoid 
sig = []

# populate sigmoid curve
for time in t_reach:
    sig.append(smoothstep(time))

# lists that will hold final time and endpoint values with their initial 
# values already inserted 
t = []
t.append(0)
endpoints_final = []
endpoints_final.append(endpoints[0])

reach_starts = []

# runs for as many iterations as there are endpoints - 1, i.e. doesn't run for
# the last one because there is no subsequent reach
for i in range(len(endpoints) - 1):
    
    # generate and populate rest period, round t values or error will start to 
    # accumulate 
    last_end_time = t[-1]
    while t[-1] < last_end_time + rest_time:
        t.append(round(t[-1] + T, 3))
        endpoints_final.append(endpoints_final[-1])
        
    reach_starts.append(t[-1])
        
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
    
# append one last rest period
last_end_time = t[-1]
while t[-1] < last_end_time + rest_time:
    t.append(round(t[-1] + T, 3))
    endpoints_final.append(endpoints_final[-1])

# write generated time and hand endpoint position data and other data related 
# constants needed later
current_experiment.t = t
current_experiment.endpoints = endpoints_final
current_experiment.rest_time = rest_time
current_experiment.reach_duration = reach_time
current_experiment.reach_times = reach_starts

# write finished results
current_model.dump()
current_experiment.dump()
