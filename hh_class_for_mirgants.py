#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Working definition of agent class (household) for ABM
 of environmental migration

@author: kelseabest
version edited by Orla O'Neill

Main differences are:
* removing the option to migrate
* making sure wealth calculates do no include land
* removing network
* stopping the posibility of employing other agents
"""

#import packages
from decisions import *
import random
import numpy as np
import pandas as pd

#object class Household
class Migrant :
    next_uid = 1
    def __init__(self, wealth_factor, ag_factor, w1, w2, w3, k, threshold): #initialize agents
        self.unique_id = Migrant.next_uid
        Migrant.next_uid += 1
        #print('hh id:',self.unique_id)

        #radomly initialize wealth
        self.wealth = random.gauss(wealth_factor, wealth_factor / 5) #adjust this for comm inequality
        self.wealth_factor = wealth_factor

        self.hh_size = np.random.poisson(5.13)
        if self.hh_size < 1:
            self.hh_size = 1
        self.individuals = pd.DataFrame() #initialize DF to hold individuals
        self.head = None
        ### set up community inequality ### 
        gini = 0.55 #gini index from BEMS is 0.55
        alpha = (1.0 / gini + 1.0) / 2.0
        self.weights = np.random.pareto(alpha)
        self.land_owned =0# self.weights*14 #np.random.lognormal(2.5, 1) #np.random.normal(14, 5) # #
        self.secure = True 
        self.wellbeing_threshold = self.hh_size * 20000 #world bank poverty threshold

        self.network_size = 10
        self.hh_network = []
        self.network_moves = 0

        self.someone_migrated = 0
        self.mig_arr=0
        self.mig_binary = 0 
        self.history = []
        self.success = []
        self.land_impacted = False
        self.wta = 0
        self.wtp = 0
        self.num_employees = 0 
        self.employees = []
        self.payments = []
        self.expenses = self.hh_size *20000 #this represents $$ to sustain HH (same as threshold)
        self.total_utility = 0
        self.total_util_w_migrant = 0
        self.num_shocked = 0
        self.ag_factor = ag_factor 
        self.land_prod = 0#self.ag_factor * self.land_owned #productivity from own land 
        
        ### TPB factors ###
        self.control = 0
        self.attitude = 0
        self.network_fact = 0
        self.weight1 = w1 / (w1 + w2 + w3) #asset weight
        self.weight2 = w2 / (w1 + w2+ w3) #experience weight
        self.weight3 = w3 / (w1 + w2 + w3) #network weight
        self.k = k 
        
        ### PMT factors #####
        self.coping_appraisal = 0
        self.threshold = threshold
        
        ### Mobility potential factors ###
        self.adaptive_capacity = 0
        self.mobility_potential = 0
        self.rootedness = random.random()
        self.unique_mig_threshold = 0
        
       # self.size_network = np.random.uniform()
        self.type='migrant'
        self.mig_angent_id=None


#assign individuals to a household
    def gather_members(self, individual_set):
        #print('migrants input to new place:',individual_set)
        ind_no_hh = individual_set#[individual_set['hh'].isnull()]
        #print('Ones with no hh:',ind_no_hh)
        if len(ind_no_hh) > self.hh_size:
            self.individuals = pd.concat([self.individuals, ind_no_hh.sample(self.hh_size)])
        else:
            self.individuals = pd.concat([self.individuals, ind_no_hh.sample(len(ind_no_hh))])
        #update information for hh and individual
        self.individuals['ind'].hh = self.unique_id
        individual_set.loc[(individual_set.id.isin(self.individuals['id'])), 'hh'] = self.unique_id
        for i in individual_set.loc[(individual_set.hh == self.unique_id), 'ind']:
            i.hh = self.unique_id
        self.individuals['hh'] = self.unique_id
       # print('migrants in new place',self.individuals)
        

    def assign_head(self, individual_set):
        my_individuals = individual_set.loc[(individual_set['hh'] == self.unique_id)]
        males = my_individuals[my_individuals['gender']== 'M']
        females = my_individuals[my_individuals['gender']== 'F']
        if (len(males) == 0 and len(females) == 0):
            head_hh = None
            return 
        elif (len(males) != 0):
            head_hh = males[males['age'] == max(males['age'])]
            self.head = head_hh
            head_hh['ind'].head = True
            #replace in individual set
            individual_set.loc[(individual_set.id.isin(head_hh['id'])), 'ind'] = head_hh
        else:
            head_hh = females[females['age'] == max(females['age'])]
            self.head = head_hh
            head_hh['ind'].head = True
            #replace in individual set
            individual_set.loc[(individual_set.id.isin(head_hh['id'])), 'ind'] = head_hh

    def check_land(self, community, comm_scale):
        if community.impacted == True:
            if random.random() < comm_scale:
                self.land_impacted = True
                self.num_shocked += 1
                self.wealth = self.wealth * random.random()
                self.land_prod = 0

    def migrate(self, method, individual_set,hh_set, mig_util, mig_threshold, community, av_wealth, av_land):
        
        return


    
    def sum_utility(self, individual_set):
        if self.unique_id not in individual_set['hh']:
            return
        my_individuals = individual_set.loc[(individual_set['hh'] == self.unique_id),'ind'].values[0]
        if isinstance(my_individuals, pd.Series):
            return
        self.total_utility = my_individuals.salary

        if self.total_utility < self.wellbeing_threshold:
            self.secure = False
        else:
            self.secure = True 

    def hire_employees(self,tf): #how many people to hire? and wtp 
        self.wtp=0
        self.num_employees=0
        self.wta = (self.wellbeing_threshold / self.hh_size) * random.random()


    def update_wealth(self, individual_set):
        if self.unique_id not in individual_set.hh:
            return
        my_individuals = individual_set.loc[(individual_set['hh'] == self.unique_id),'ind'].values[0] #takes all individuals in upazila but then filters them off to one hh here
        if isinstance(my_individuals, pd.Series):
            return
        
        self.wealth = self.wealth + my_individuals.salary - self.expenses - np.sum(self.payments) + self.land_prod
        
        if self.wealth < 0:
            self.wealth = 0 
            self.secure = False 

        #reset these values
        self.land_impacted = False

    def check_network(self, hh_set):
        self.network_moves = 0
