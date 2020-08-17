# -*- coding: utf-8 -*-
"""
Generates compeletely fresh model and experiment object and pickle dumps them
in the working directory
@author: Jack Vincent
"""

import model

model_name = 'upper_arm_0'
experiment_name = '8-17-20'

model.init_model(model_name)
model.init_experiment(experiment_name)
