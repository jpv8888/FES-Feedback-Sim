# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 00:01:11 2020

@author: 17049
"""

import model

current_model = model.load('upper_arm_0')
current_experiment = model.load('8-17-20')
current_muscle_tracker = model.load('test')

mode = 'constant'
constant_excites = [0.3, 0.1, 0, 0, 0]

if mode == 'constant':
    t = current_muscle_tracker.t
    n_samples = len(t)
    for i, muscle_data in enumerate(current_muscle_tracker.muscles):
        muscle_data.forward_excite = [constant_excites[i]] * n_samples
        
# write finished results
current_model.dump()
current_experiment.dump()
current_muscle_tracker.dump()