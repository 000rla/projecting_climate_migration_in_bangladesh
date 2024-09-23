#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Working definition of decision class for ABM
 of environmental migration

@author: kelseabest
"""

#import packages
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class decision :
    #method decide returns True or False
    #subclass of decisions
    def __init__(self): #initialize outcome
        self.outcome = False
    def decide(self):
        pass

class utility_max(decision):
    def __init__(self): #initialize utilities
        super().__init__()
    def decide(self, household):
        if household.total_utility < household.total_util_w_migrant:
            self.outcome = True

class utility_return_time(decision):
    def __init__(self):
        super().__init__()
    def decide(self, household):
        return_time = 1
        if (household.total_util_w_migrant - household.total_utility)*return_time - household.mig_cost >= 0:
            self.outcome = True
        
        
class push_threshold(decision):
    def __init__(self): #initialize utilities
        super().__init__()
    def decide(self, household):
        if household.secure == False:
            self.outcome = True 
        elif household.total_utility < household.total_util_w_migrant:
            self.outcome = True

class tpb(decision):
    def __init__(self): #initialize utilities
        super().__init__()
    def decide(self, household):
        if random.random() <= household.control:
            perceived_control = 1
        else:
            perceived_control = 0 
        if random.random() <= household.attitude * household.network_fact:
            attitude_scaled = 1
        else:
            attitude_scaled = 0 
        if perceived_control == 1 and attitude_scaled == 1:
            self.outcome = True

class pmt(decision):
    def __init__(self): #initialize utilities
        super().__init__()
    def decide(self, household):
        if random.random() <= household.coping_appraisal:
            self.outcome = True

class mobility_potential(decision):
    def __init__(self): #initialize utilities
        super().__init__()
    def decide(self, household):
        if random.random() <= household.unique_mig_threshold:
            self.outcome = True

class hybrid(decision):
    def __init__(self): #initialize utilities
        super().__init__()
    def decide(self, household):
        if random.random() <= household.control:
            perceived_control = 1
        else:
            perceived_control = 0 
        if random.random() <= household.attitude * household.network_fact:
            attitude_scaled = 1
        else:
            attitude_scaled = 0 
        if perceived_control == 1 and attitude_scaled == 1:
            self.outcome = True