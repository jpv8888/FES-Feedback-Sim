# -*- coding: utf-8 -*-
"""
FD experimental

@author: 17049
"""

from math import radians
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

import model

sim_name = 'test'
f_s = 100

# 2 modes: 'muscles' and 'ideal'
drive = 'muscles'

# 2 modes: 'only_reaches' and 'free_run'
mode = 'free_run'

# 2 modes: 'from_ID' and 'from_ANN' (where muscle excitations come from)
excite_mode = 'from_ANN'

# used only for 'free_run'
duration = 10

# load in model and experiment
current_model = model.load('upper_arm_0')
current_experiment = model.load('8-17-20')
current_muscle_tracker = model.load('test')

current_simulation = model.Simulation(sim_name, current_model, f_s)

# whether or not to manually set the model's initial joint angles, if false, 
# initial joint angles will be obtained from the loaded experiment
manual_init = True

init_joint_angles = []
for joint_data in current_experiment.joints:
    init_joint_angles.append(joint_data.angle[0])
    
manual_init_joint_angles = [1.5708, 3.1416]

for i, joint_data in enumerate(current_simulation.joints):
    if manual_init == True:
        joint_data.angle.append(manual_init_joint_angles[i])
    elif manual_init == False:
        joint_data.angle.append(init_joint_angles[i])
    
# %% Convert excitations to activations

for muscle_data in current_muscle_tracker.muscles:
    
    if excite_mode == 'from_ID':    
        muscle_data.forward_excite = muscle_data.excitation
        
    elif excite_mode == 'from_ANN':
        # externally produced muscle excitations are written in another script
        pass

t = current_muscle_tracker.t
tau = 0.01
for muscle_data in current_muscle_tracker.muscles:
    
    muscle_data.forward_excite_interp = UnivariateSpline(t, muscle_data.forward_excite, s=0)
    
    def act_derivative(time, act, tau, muscle_data):
        u = muscle_data.forward_excite_interp(time)
        dadt = (u - act)/tau
        return dadt
    
    tspan = t
    xinit = [muscle_data.activation[0]]
    sol = solve_ivp(act_derivative, [tspan[0], tspan[-1]], xinit, t_eval=tspan,
                    args=(tau, muscle_data), rtol=1e-4)
    
    muscle_data.forward_act = list(sol.y[0])
    muscle_data.forward_act_interp = UnivariateSpline(t, muscle_data.forward_act, s=0)
    
# %% Define derivative function

if drive == 'ideal':
    def f1(current_model, current_experiment, joint_index, x_vec, t):
        current_model.skeleton.write_joint_angles(x_vec)
        current_model.skeleton.calc_I()
        gravity = current_model.skeleton.calc_gravity()[joint_index]
        I = current_model.skeleton.joints[joint_index].I
        torque = current_experiment.return_torques(t)[joint_index]
        return (torque + gravity)/I

elif drive == 'muscles':
    def f1(current_model, current_experiment, joint_index, x_vec, t):
        current_model.skeleton.write_joint_angles(x_vec)
        current_model.skeleton.calc_I()
        gravity = current_model.skeleton.calc_gravity()[joint_index]
        I = current_model.skeleton.joints[joint_index].I
        torque = current_muscle_tracker.return_torques(t)[joint_index]
        return (torque + gravity)/I
    
def f2(t, x, current_model, current_experiment):
    dxdt = []
    dxdt.append(x[1])
    dxdt.append(f1(current_model, current_experiment, 0, [x[0], x[2]], t))
    dxdt.append(x[3])
    dxdt.append(f1(current_model, current_experiment, 1, [x[0], x[2]], t))
    return dxdt

if mode == 'only_reaches':
    
    for reach_start in current_experiment.reach_times:
        current_simulation.t.extend(list(np.arange(current_simulation.t[-1], reach_start - 1/f_s, 1/f_s)))
        rest_samples = current_experiment.rest_time * f_s
        for j, joint_data in enumerate(current_simulation.joints):
            joint_data.angle.extend([float(current_experiment.joints[j].angle_interp(reach_start))] * (rest_samples))
        start_time = reach_start
        tspan = np.arange(start_time, start_time + current_experiment.reach_duration, 1/f_s)
        current_simulation.t.extend(list(tspan))
        xinit = [current_simulation.joints[0].angle[-1], 0, current_simulation.joints[1].angle[-1], 0]
        sol = solve_ivp(f2, [tspan[0], tspan[-1]], xinit, t_eval=tspan, method='Radau', 
                        args=(current_model, current_experiment), rtol=1e-4, atol=1e-10)
        current_simulation.joints[0].angle.extend(list(sol.y[0]))
        current_simulation.joints[1].angle.extend(list(sol.y[2]))
        
    current_simulation.t.extend(list(np.arange(current_simulation.t[-1], current_experiment.t[-1], 1/f_s)))
    rest_samples = current_experiment.rest_time * f_s
    for j, joint_data in enumerate(current_simulation.joints):
        joint_data.angle.extend([float(current_simulation.joints[j].angle[-1])] * rest_samples)
        
    del current_simulation.t[0]
    
    anim = current_model.animate(current_simulation)
    
elif mode == 'free_run':
    
    # starting at 0 can cause really long computation times; I believe because
    # the model must extrapolate many times into negative time in order to 
    # calculate the first few derivatives, so we start at the onset of the
    # first reach
    start_time = current_experiment.reach_times[0]
    tspan = np.arange(start_time, start_time + duration, 1/f_s)
    current_simulation.t.extend(list(tspan))
    xinit = [current_simulation.joints[0].angle[0], 0, current_simulation.joints[1].angle[0], 0]
    sol = solve_ivp(f2, [tspan[0], tspan[-1]], xinit, t_eval=tspan, method='BDF', 
                args=(current_model, current_experiment), rtol=1e-4, atol=1e-10)
    current_simulation.joints[0].angle.extend(list(sol.y[0]))
    current_simulation.joints[1].angle.extend(list(sol.y[2]))
    
    anim = current_model.animate(current_simulation)


    
    
    



    
    