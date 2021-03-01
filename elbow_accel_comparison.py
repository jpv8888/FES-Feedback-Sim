# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 21:59:40 2021

@author: 17049
"""

from math import radians
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline


import model

# load in model and experiment
current_model = model.load('upper_arm_0')
current_experiment = model.load('8-17-20')
current_muscle_tracker = model.load('test')

t = current_experiment.t

joint_data_shoulder = current_experiment.joints[0]
elbow_accel_from_angles = []

test = []
test2 = []
for time in t:
    
    x_vec = []
    for joint_data in current_experiment.joints:
        x_vec.append(joint_data.angle_interp(time))
    current_model.skeleton.write_joint_angles(x_vec)
    current_model.skeleton.calc_I()
    
    alpha_shoulder = joint_data_shoulder.angle_interp.derivative(n=2)(time)
    omega_shoulder = joint_data_shoulder.angle_interp.derivative(n=1)(time)
    l_humerus = current_model.skeleton.bones[1].length
    alpha_tan_mag = abs(alpha_shoulder * l_humerus)
    alpha_cen_mag = abs((omega_shoulder**2) * l_humerus)  
    
    # 180 degree rotation, centripetal acceleration vector runs from elbow 
    # to shoulder
    cen_vector = [i*-1 for i in current_model.skeleton.joints[1].location]
    cen_vector_mag = l_humerus
    cen_vector_unit = [i/cen_vector_mag for i in cen_vector]
    
    tan_vector_unit = [0, 0]
    if alpha_shoulder >= 0:
        tan_vector_unit[0] = -1 * cen_vector_unit[1]
        tan_vector_unit[1] = cen_vector_unit[0]
    else:
        tan_vector_unit[0] = cen_vector_unit[1]
        tan_vector_unit[1] = -1 * cen_vector_unit[0]
        
    cen_vector_final = [i*alpha_cen_mag for i in cen_vector_unit]
    tan_vector_final = [i*alpha_tan_mag for i in tan_vector_unit]
    test.append(tan_vector_final)
    elbow_lin_accel = [x + y for x, y in zip(cen_vector_final, tan_vector_final)]
    elbow_accel_from_angles.append(elbow_lin_accel)
    
elbow_accel_from_angles_x = [i[0] for i in elbow_accel_from_angles]
elbow_accel_from_angles_y = [i[1] for i in elbow_accel_from_angles]

a_x = []
a_y = []
for time in t:
    a_x.append(current_experiment.elbow_loc_interp_x.derivative(n=2)(time))
    a_y.append(current_experiment.elbow_loc_interp_y.derivative(n=2)(time))
    
plt.plot(t, elbow_accel_from_angles_x, label='from angles')
plt.plot(t, a_x, label='straightforward')
plt.legend()

plt.plot(t, elbow_accel_from_angles_y, label='from angles')
plt.plot(t, a_y, label='straightforward')
plt.legend()

# %% testing

t = current_experiment.t
joint_index = 0

angle_interp_test = UnivariateSpline(t, current_experiment.joints[joint_index].angle, s=0)
angle_interp_test_list = []
for time in t:
    angle_interp_test_list.append(angle_interp_test.derivative(n=2)(time))
    
gradient = np.gradient(current_experiment.joints[joint_index].angle, 0.001)
gradient = np.gradient(gradient, 0.001)
    
plt.plot(t, angle_interp_test_list, color='r')
plt.plot(t, gradient, color='g')

# %%

# green should be rotated 90 degrees CCW, blue 90 degrees CW
vector = [-0.5, -0.75]
fig, ax = plt.subplots()
ax.axis('square')
ax.arrow(0, 0, vector[0], vector[1], head_width=0.2, color='r')
ax.arrow(0, 0, -1*vector[1], vector[0], head_width=0.2, color='g')
ax.arrow(0, 0, vector[1], -1*vector[0], head_width=0.2, color='b')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
