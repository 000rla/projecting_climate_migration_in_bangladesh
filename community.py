#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Working definition of community class for ABM
 of environmental migration

@author: kelseabest
version edited by Orla O'Neill
"""

#import packages
#import random
#import math
#import numpy as np
#import matplotlib.pyplot as plt

class community :
    def __init__(self, n_hh, n_jobs, comm_impact,F,C,H,ag_factor):
        self.impacted = False
        self.n_hh = n_hh   
        self.avail_jobs = n_jobs
        self.num_impacted = 0
        self.comm_impact = comm_impact
        self.F=F #floods
        self.C=C #cyclones
        self.H=H #heatwaves
        self.ag_factor=ag_factor
        
        #environmental shock
    def shock(self,time):
        self.weather=False

        #if there is a heatwave
        if self.H.sel(time=time).ts.values<0:
            factor=self.H.sel(time=time).values*4.96 #number from Heinicke et al 2022 
            self.impacted = True
            self.weather=True
            self.num_impacted += 1
            self.avail_jobs = self.avail_jobs * (1 - self.comm_impact*self.H.sel(time=time).values)#number of jobs decreases with scale of community impact
            self.ag_factor = self.ag_factor * (100-factor)/100 

        #if there is a flood
        if self.F.sel(time=time).I_flood.values<0:
            factor=self.F.sel(time=time).values*10.83 #number from Hasan et al 2021
            self.impacted = True
            self.weather=True
            self.num_impacted += 1
            self.avail_jobs = self.avail_jobs * (1 - self.comm_impact*self.F.sel(time=time).values)  #number of jobs decreases with scale of community impact
            self.ag_factor = self.ag_factor * (100-factor)/100 
            # print(self.ag_factor)

        #if there is a cyclone
        if self.C.sel(time=time).wind_speed.values<0:
            factor=self.C.sel(time=time).values*10.83 #number from Hasan et al 2021
            self.impacted = True
            self.weather=True
            self.num_impacted += 1
            self.avail_jobs = self.avail_jobs * (1 - self.comm_impact*self.C.sel(time=time).values)  #number of jobs decreases with scale of community impact
            self.ag_factor = self.ag_factor * (100-factor)/100 
            

#origin community
class origin(community):
    def __init__(self, n_hh, n_jobs, comm_impact,F,C,H,ag_factor):
        super(origin, self).__init__(n_hh, n_jobs, comm_impact,F,C,H,ag_factor)
        self.weather=False
    def shock(self,time):
        super(origin, self).shock(time)

#destinations
class dhaka(community):
    def __init__(self):
        super().__init__()

class khulna(community):
    def __init__(self):
        super().__init__()

class rural(community):
    def __init__(self):
        super().__init__()
