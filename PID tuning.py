# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 08:31:14 2021

@author: 17049
"""

# -*- coding: utf-8 -*-
"""
FD experimental

@author: 17049
"""
test2 = []

import copy
from math import radians
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import hyperopt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline

import model

sim_name = 'test'
f_s = 1000

# 2 modes: 'muscles' and 'ideal'
drive = 'muscles'

# 2 modes: 'only_reaches' and 'free_run', 'free_run_with_feedback'
mode = 'free_run_with_feedback'

# 2 modes: 'from_ID' and 'from_ANN' (where muscle excitations come from)
excite_mode = 'from_ID'

# used only for 'free_run'
duration = 10

# load in model and experiment
current_model = model.load('upper_arm_0')
current_experiment = model.load('8-17-20')
current_muscle_tracker = model.load('test')

current_simulation = model.Simulation(sim_name, current_model, f_s)

# whether or not to manually set the model's initial joint angles, if false, 
# initial joint angles will be obtained from the loaded experiment
manual_init = False

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
        torque = current_muscle_tracker.return_torques(t, x_vec)[joint_index]
        return (torque + gravity)/I
    
    def f_elbow(current_model, current_experiment, joint_index, x_vec, t, 
                omega_shoulder, alpha_shoulder):
        current_model.skeleton.write_joint_angles(x_vec)
        current_model.skeleton.calc_I()
        gravity = current_model.skeleton.calc_gravity()[joint_index]
        I = current_model.skeleton.joints[joint_index].I
        torque = current_muscle_tracker.return_torques(t, x_vec)[joint_index]
        
        l_humerus = current_model.skeleton.bones[1].length
        alpha_tan_mag = abs(alpha_shoulder * l_humerus)
        alpha_cen_mag = abs((omega_shoulder**2) * l_humerus)
        
        # 180 degree rotation, centripetal acceleration vector runs from elbow 
        # to shoulder
        cen_vector = [i*-1 for i in current_model.skeleton.bones[2].endpoint1.coords]
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
        elbow_lin_accel = [x + y for x, y in zip(cen_vector_final, tan_vector_final)]
        
        elbow_coords = current_model.skeleton.joints[1].location
        if current_model.skeleton.bones[2].point_mass[0] != 0:
            CoM_coords = current_model.skeleton.bones[2].CoM_with_PM
        else:
            CoM_coords = current_model.skeleton.bones[2].CoM
        rho_vec = [x - y for x, y in zip(CoM_coords, elbow_coords)]
        
        rho_x = rho_vec[0]
        rho_y = rho_vec[1]
        a_x = elbow_lin_accel[0]
        a_y = elbow_lin_accel[1]
        
        if current_model.skeleton.bones[2].point_mass[0] != 0:
            point_mass = current_model.skeleton.bones[2].point_mass[0]
            forearm_mass = current_model.skeleton.bones[2].mass + point_mass
        else:
            forearm_mass = current_model.skeleton.bones[2].mass
        
        inertial_torque = (rho_x*forearm_mass*a_y) - (rho_y*forearm_mass*a_x)
        
        # return (torque + gravity)/I
        return (torque + gravity - inertial_torque)/I
    
    
def f2(t, x, current_model, current_experiment):
    omega_shoulder = x[1]
    alpha_shoulder = f1(current_model, current_experiment, 0, [x[0], x[2]], t)
    omega_elbow = x[3]
    alpha_elbow = f_elbow(current_model, current_experiment, 1, [x[0], x[2]], 
                          t, omega_shoulder, alpha_shoulder)
    dxdt = []
    dxdt.append(omega_shoulder)
    dxdt.append(alpha_shoulder)
    dxdt.append(omega_elbow)
    dxdt.append(alpha_elbow)
    return dxdt

# Tuning parameters: proportional, integral, and derivative gain
Kp = 0.01
Ki = 0.01
Kd = 0.01

# how many samples to forward integrate before updating the muscle activations
# in other words, how often to provide feedback
samples_before_update = 5

def elbow_feedback(current_sample_idx, err, err_i, err_d, Kp, Ki, Kd):
    elbow_flexor_indices = [5, 6]
    for idx in elbow_flexor_indices:
        acts_to_be_adjusted = current_muscle_tracker.muscles[idx].forward_act[current_sample_idx:]
        adjusted_acts = [el - Kp*err for el in acts_to_be_adjusted]
        adjusted_acts = [el - Ki*err_i for el in acts_to_be_adjusted]
        adjusted_acts = [el - Kd*err_d for el in acts_to_be_adjusted]
        current_muscle_tracker.muscles[idx].forward_act[current_sample_idx:] = []
        current_muscle_tracker.muscles[idx].forward_act.extend(adjusted_acts)
        current_muscle_tracker.muscles[idx].forward_act_interp = UnivariateSpline(t, current_muscle_tracker.muscles[idx].forward_act, s=0)
           
    elbow_extensor_indices = [3]
    for idx in elbow_extensor_indices:
        acts_to_be_adjusted = current_muscle_tracker.muscles[idx].forward_act[current_sample_idx:]
        adjusted_acts = [el + Kp*err for el in acts_to_be_adjusted]
        adjusted_acts = [el + Ki*err_i for el in acts_to_be_adjusted]
        adjusted_acts = [el + Kd*err_d for el in acts_to_be_adjusted]
        current_muscle_tracker.muscles[idx].forward_act[current_sample_idx:] = []
        current_muscle_tracker.muscles[idx].forward_act.extend(adjusted_acts)
        current_muscle_tracker.muscles[idx].forward_act_interp = UnivariateSpline(t, current_muscle_tracker.muscles[idx].forward_act, s=0)

def shoulder_feedback(current_sample_idx, err, err_i, err_d, Kp, Ki, Kd):
    shoulder_flexor_indices = [0]
    for idx in shoulder_flexor_indices:
        acts_to_be_adjusted = current_muscle_tracker.muscles[idx].forward_act[current_sample_idx:]
        adjusted_acts = [el + Kp*err for el in acts_to_be_adjusted]
        adjusted_acts = [el + Ki*err_i for el in acts_to_be_adjusted]
        adjusted_acts = [el + Kd*err_d for el in acts_to_be_adjusted]
        current_muscle_tracker.muscles[idx].forward_act[current_sample_idx:] = []
        current_muscle_tracker.muscles[idx].forward_act.extend(adjusted_acts)
        current_muscle_tracker.muscles[idx].forward_act_interp = UnivariateSpline(t, current_muscle_tracker.muscles[idx].forward_act, s=0)
           
    shoulder_extensor_indices = [1]
    for idx in shoulder_extensor_indices:
        acts_to_be_adjusted = current_muscle_tracker.muscles[idx].forward_act[current_sample_idx:]
        adjusted_acts = [el - Kp*err for el in acts_to_be_adjusted]
        adjusted_acts = [el - Ki*err_i for el in acts_to_be_adjusted]
        adjusted_acts = [el - Kd*err_d for el in acts_to_be_adjusted]
        current_muscle_tracker.muscles[idx].forward_act[current_sample_idx:] = []
        current_muscle_tracker.muscles[idx].forward_act.extend(adjusted_acts)
        current_muscle_tracker.muscles[idx].forward_act_interp = UnivariateSpline(t, current_muscle_tracker.muscles[idx].forward_act, s=0)
        

    

    

def objective(params):
    Kp, Ki, Kd = params['Kp'], params['Ki'], params['Kd']
    current_simulation = model.Simulation(sim_name, current_model, f_s)
    current_muscle_tracker = model.load('test')
    
    # whether or not to manually set the model's initial joint angles, if false, 
    # initial joint angles will be obtained from the loaded experiment
    manual_init = False
    
    init_joint_angles = []
    for joint_data in current_experiment.joints:
        init_joint_angles.append(joint_data.angle[0])
        
    manual_init_joint_angles = [1.5708, 3.1416]
    
    for i, joint_data in enumerate(current_simulation.joints):
        if manual_init == True:
            joint_data.angle.append(manual_init_joint_angles[i])
        elif manual_init == False:
            joint_data.angle.append(init_joint_angles[i])
    
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
    
    
    start_time = current_experiment.reach_times[0]
    tspan = np.arange(0, start_time, 1/f_s)
    current_simulation.t.extend(list(tspan))
    
    # writes the values of the joint angles throughout the rest period
    rest_samples = current_experiment.rest_time * f_s
    for j, joint_data in enumerate(current_simulation.joints):
        joint_data.angle.extend([float(current_experiment.joints[j].angle_interp(start_time))] * (rest_samples))
    
    current_sample_idx = int(start_time * f_s)
    
    feedback_period = samples_before_update/f_s
    shoulder_previous_error = 0
    shoulder_error_integral = 0
    shoulder_error_derivative = 0
    elbow_previous_error = 0
    elbow_error_integral = 0
    elbow_error_derivative = 0
    
    
    total_joint_error = 0
    while current_sample_idx < (len(t) - 1):
        
        start_time = t[current_sample_idx]
        tspan = np.linspace(start_time, t[current_sample_idx + samples_before_update], num=5)
        current_simulation.t.extend(list(tspan))
        
        # initial guesses
        xinit = [current_simulation.joints[0].angle[-1], 
                 current_simulation.joints[0].velocity[-1], 
                 current_simulation.joints[1].angle[-1], 
                 current_simulation.joints[1].velocity[-1]]
        
        sol = solve_ivp(f2, [tspan[0], tspan[-1]], xinit, t_eval=tspan, method='Radau', 
                        args=(current_model, current_experiment), rtol=1e-4, atol=1e-10)
        current_simulation.joints[0].angle.extend(list(sol.y[0]))
        current_simulation.joints[0].velocity.extend(list(sol.y[1]))
        current_simulation.joints[1].angle.extend(list(sol.y[2]))
        current_simulation.joints[1].velocity.extend(list(sol.y[3]))
        
        current_sample_idx += samples_before_update
        print(current_sample_idx)
        
        
        # shoulder feedback
        shoulder_error = current_experiment.joints[0].angle[current_sample_idx] - current_simulation.joints[0].angle[-1]
        shoulder_error_integral += shoulder_error
        shoulder_error_derivative = (shoulder_error - shoulder_previous_error)/feedback_period
        shoulder_feedback(current_sample_idx, shoulder_error, 
                          shoulder_error_integral, shoulder_error_derivative, 
                          Kp, Ki, Kd)
            
        # elbow feedback
        elbow_error = current_experiment.joints[1].angle[current_sample_idx] - current_simulation.joints[1].angle[-1]
        elbow_error_integral += elbow_error
        elbow_error_derivative = (elbow_error - elbow_previous_error)/feedback_period
        elbow_feedback(current_sample_idx, elbow_error, elbow_error_integral, 
                       elbow_error_derivative, Kp, Ki, Kd)
        
        shoulder_previous_error = shoulder_error
        elbow_previous_error = elbow_error
        
        current_muscle_tracker.trim_acts()
        

        for j, joint_data in enumerate(current_experiment.joints):
            total_joint_error += abs(joint_data.angle[current_sample_idx] - current_simulation.joints[j].angle[current_sample_idx])
            
        if total_joint_error > 1000:
            total_joint_error = 1000 + (len(t) - current_sample_idx)
            break
        
    
    return total_joint_error
            
    
space = {'Kp': hp.uniform('Kp', 0.68, 1.18),
         'Ki': hp.uniform('Ki', 0.2, 1.2),
         'Kd': hp.uniform('Kd', 0, 0.9)}

trials = Trials()

best = fmin(
    fn=objective, # Objective Function to optimize
    space=space, # Hyperparameter's Search Space
    algo=tpe.suggest, # Optimization algorithm (representative TPE)
    max_evals=100, # Number of optimization attempts
    trials=trials
)

    
        
