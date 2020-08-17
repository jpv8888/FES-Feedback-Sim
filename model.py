# -*- coding: utf-8 -*-
"""
Classes and functions for generating a completed model and coordinating 
actions that require crosstalk between skeletal and muscular information
@author: Jack Vincent
"""

import pickle

from bones import Bone, Joint, Skeleton

# writes and dumps a fresh model in the working directory
def init_model(name):
    
    # name for this particular model instance
    name = name
    
    # for bone lengths:
    # Pan N. Length of long bones and their proportion to body height in 
    # hindus. J Anat. 1924;58:374-378. 
    # https://pubmed.ncbi.nlm.nih.gov/17104032 
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1249729/.
    
    # for segment masses, mass information is stored in bones, but is assumed
    # to also account for mass of muscle, connective tissue, etc.:
    # DA Winter, Biomechanics and Motor Control of Human Movement, 3rd edition 
    # (John Wiley & Sons 2005)
    
    # units are MKS
    height = 1.7
    mass = 65
    
    # "scapula" is a major abstraction, mass is irrelevant, and length is
    # irrelevant beyond giving sufficient room for muscle attachment sites, so 
    # these dimensions are mostly arbitrary 
    scapula = Bone('scapula', 0.05*height, 0.01*mass, False, False)
    
    # generate other bones
    humerus = Bone('humerus', 0.188*height, 0.028*mass, False, True)
    radioulna  = Bone('radioulna', 0.164*height, 0.022*mass, True, True)
    
    # initial bone endpoints, shoulder joint is always fixed and defined as 
    # the origin 
    endpoints_0 = {'scapula': ([scapula.length, 0], [0, 0]),
                   'humerus': ([0, 0], [0, -humerus.length]), 
                   'radioulna': ([0, -humerus.length], 
                                 [0, -humerus.length-radioulna.length])}
    
    # generate joints
    shoulder = Joint('shoulder', 'scapula', 'humerus', 60, 210, 90, 'cw')
    elbow = Joint('elbow', 'humerus', 'radioulna', 30, 180, 180, 'ccw')
    
    # generate fresh, completed skeleton
    skeleton = Skeleton([scapula, humerus, radioulna], [shoulder, elbow], 
                        endpoints_0)
    
    # dump model
    with open(name, "wb") as fp:
        pickle.dump(Model(name, skeleton), fp)
        
# writes and dumps a fresh experiment in the working directory
def init_experiment(name):
    
    # dump experiment
    with open(name, "wb") as fp:
        pickle.dump(Experiment(name), fp)
        
# load in a model or experiment from the working directory
def load(name):
    
    # read in model or experiment
    with open(name, "rb") as fp:   
        return(pickle.load(fp))

# model object composed of skeleton and musculature 
class Model:
    
    def __init__(self, name, skeleton):
        
        self.name = name
        self.skeleton = skeleton
    
    # store self in working directory using pickle
    def dump(self):
        with open(self.name, "wb") as fp:
            pickle.dump(self, fp)
       
    # necessary to be able to pickle this object
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    
# experiment object, keeps track of data through various processing scripts
class Experiment:
    
    def __init__(self, name):
        
        self.name = name
        self.t = []
        self.endpoints = []
        self.joints = []
        
    # store self in working directory using pickle
    def dump(self):
        with open(self.name, "wb") as fp:
            pickle.dump(self, fp)
        
    # necessary to be able to pickle this object
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    
# tracking data related to a single joint through an experiment
class JointData:
    
    def __init__(self, name):
    
        self.name = name
        self.angle = []

