#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Working definition of individual class for ABM
 of environmental migration

@author: kelseabest
"""

#import packages
from decisions import *
import numpy as np
import pandas as pd

class Individual :
    next_uid = 1

    def __init__(self, ag_factor,ID): #initialize
        self.unique_id = Individual.next_uid
        Individual.next_uid += 1
        self.age = np.random.weibull(1.68) * 33.6
        gend_arr = ['M', 'F']
        self.gender = np.random.choice(gend_arr, 1)
        self.hh = None
        self.employment = None
        self.salary = 0
        self.employer = None
        self.can_migrate  = False
        self.head = False
        self.migrated = False
        self.ag_factor = ag_factor
        self.alive = True 
        self.wta = 0 
        self.originally_from=ID
        self.currently_living=ID
        self.mig_dest=None
        #self.mig_id=None


    def age_up(self):
        self.age = self.age + 1
        self.salary = 0 

    def check_eligibility(self):
        #is the agent eligible to migrate?
        if 35> self.age >= 14 and self.gender == 'M' and self.migrated == False:
            self.can_migrate = True

        #individuals look for work within community
    def find_work(self, hh_set, mig_util): 
        #look for ag in own land first
        #util_migrate = mig_util #global var
        my_hh = hh_set[hh_set['hh_id'] == self.hh]
        
        if self.hh == None:
            return
        
        else:
            if my_hh.empty:
                print('hh is empty')
                print('looking for hh:',self.hh)
                print(hh_set)
            my_house = my_hh.loc[0,'household']
            
            if type(my_house)==pd.Series:
                my_house=my_house.iloc[0]
        
        #too young to work?
        if self.age < 14 or self.gender != 'M' or self.age>70: #should this be "or"?
            self.employment = 'None'
            self.salary = 0
        #work in ag on own land
        elif my_house.land_impacted == False and my_house.land_owned > 20 and my_hh.type[0]!='migrant': #making this 100 for new land dist, previously 20
            self.employment = "SelfAg"
            self.salary = my_house.land_owned * self.ag_factor * 2 #/ my_house.hh_size 

        else:
            self.employment = "Looking" 
            self.wta = my_house.wta
            self.salary = 0  
