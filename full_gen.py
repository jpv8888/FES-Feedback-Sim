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
f_s = 100

# generate and save new model and experiment
model.init_model(model_name)
model.init_experiment(experiment_name, f_s)

# load in model and experiment
current_model = model.load(model_name)
current_experiment = model.load(experiment_name)
