# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 08:45:32 2020

@author: 17049
"""
from bones import *

def init():
    
    # for bone lengths:
    # Pan N. Length of long bones and their proportion to body height in 
    # hindus. J Anat. 1924;58:374-378. https://pubmed.ncbi.nlm.nih.gov/17104032 
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
    
    humerus = Bone('humerus', 0.188*height, 0.028*mass, False, True)
    radioulna  = Bone('radioulna', 0.164*height, 0.022*mass, True, True)
    
    # shoulder joint is always fixed and defined as the origin 
    endpoints_0 = {'scapula': ([scapula.length, 0], [0, 0]),
                   'humerus': ([0, 0], [0, -humerus.length]), 
                   'radioulna': ([0, -humerus.length], 
                                 [0, -humerus.length-radioulna.length])}
        
    shoulder = Joint('shoulder', 'scapula', 'humerus', 60, 210, 90, 'cw')
    elbow = Joint('elbow', 'humerus', 'radioulna', 30, 180, 180, 'ccw')
    
    skeleton = Skeleton([scapula, humerus, radioulna], [shoulder, elbow], 
                        endpoints_0)
    
    return skeleton