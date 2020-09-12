# -*- coding: utf-8 -*-
"""
Generates compeletely fresh model and experiment object and pickle dumps them
in the working directory
@author: Jack Vincent
"""

import model

# model and experiment parameters
model_name = 'upper_arm_0'
experiment_name = '8-17-20'
f_s = 1000

# generate and save new model 
model.init_model(model_name)
current_model = model.load(model_name)

# generate and save new experiment
model.init_experiment(experiment_name, f_s, current_model)
current_experiment = model.load(experiment_name)
