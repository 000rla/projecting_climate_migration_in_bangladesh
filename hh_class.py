#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Working definition of agent class (household) for ABM
 of environmental migration

@author: kelseabest
"""

#import packages
from decisions import *
import random
import numpy as np
import pandas as pd
import networkx as nx

#object class Household
class Household :
    next_uid = 1
    def __init__(self, wealth_factor, ag_factor, w1, w2, w3, k, threshold,size): #initialize agents
        self.unique_id = Household.next_uid
        Household.next_uid += 1
        #print('hh id:',self.unique_id)

        #radomly initialize wealth
        self.wealth = random.gauss(wealth_factor, wealth_factor / 5) #adjust this for comm inequality
        self.wealth_factor = wealth_factor

        self.hh_size = np.random.poisson(size) #change to no people/no hhs?
        if self.hh_size < 1:
            self.hh_size = 1
        self.individuals = pd.DataFrame() #initialize DF to hold individuals
        self.head = None
        ### set up community inequality ### 
        gini = 0.55 #gini index from BEMS is 0.55
        alpha = (1.0 / gini + 1.0) / 2.0
        self.weights = np.random.pareto(alpha)
        self.land_owned = self.weights*14 #np.random.lognormal(2.5, 1) #np.random.normal(14, 5) # #
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
        self.land_prod = self.ag_factor * self.land_owned #productivity from own land 
        
        ### TPB factors ###
        self.control = 0
        self.attitude = 0
        self.network_fact = 0
        self.weight1 = w1 / (w1 + w2 + w3) #asset weight
        self.weight2 = w2 / (w1 + w2+ w3) #experience weight
        self.weight3 = w3 / (w1 + w2 + w3) #network weight
        self.k = k #shape parameter for one of the TPB components.
        
        ### PMT factors #####
        self.coping_appraisal = 0
        self.threshold = threshold
        
        ### Mobility potential factors ###
        self.adaptive_capacity = 0
        self.mobility_potential = 0
        self.rootedness = random.random()
        self.unique_mig_threshold = 0
        
        # self.size_network = np.random.uniform()
        self.mig_angent_id=None
        self.type='normal'

#assign individuals to a household
    def gather_members(self, individual_set):
        ind_no_hh = individual_set[individual_set['hh'].isnull()]
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
        self.mig_angent_id=None
        util_migrate = mig_util
        self.mig_cost = mig_threshold

        my_individuals = individual_set.loc[(individual_set['hh'] == self.unique_id, 'ind')]
        can_migrate = []
        for i in my_individuals:
            if i is not None:
                if i.can_migrate == True and i.migrated == False:
                    can_migrate.append(i)
        if len(can_migrate) != 0:
            migrant = np.random.choice(can_migrate, 1)
            #print(migrant[0])
        else:
            #print('lost at first else')
            return

        if method == 'utility' and self.wealth > mig_threshold:
            self.total_util_w_migrant = self.total_utility - migrant[0].salary + util_migrate 
            decision = utility_max()
            decision.decide(self)
            if decision.outcome == True:
                self.wealth = self.wealth - mig_threshold #subtract out mig_threshold cost
                self.someone_migrated += 1
                self.mig_binary = 1
                #migrant[0].migrated = True
                migrant[0].salary = util_migrate
                individual_set.loc[(individual_set.id == migrant[0].unique_id), 'ind'] = migrant[0]
                
                
        elif method == 'utility_return_time' and self.wealth > mig_threshold:
            self.total_util_w_migrant = self.total_utility - migrant[0].salary + util_migrate 
            decision = utility_return_time()
            decision.decide(self)
            if decision.outcome == True:
                self.wealth = self.wealth - mig_threshold #subtract out mig_threshold cost
                self.someone_migrated += 1
                self.mig_binary = 1
                #migrant[0].migrated = True
                migrant[0].salary = util_migrate
                individual_set.loc[(individual_set.id == migrant[0].unique_id), 'ind'] = migrant[0]


        elif method == 'push_threshold' and self.wealth > mig_threshold:
            self.total_util_w_migrant = self.total_utility - migrant[0].salary + util_migrate 
            decision = push_threshold()
            decision.decide(self)
            if decision.outcome == True:
                self.wealth = self.wealth - mig_threshold #subtract out mig_threshold cost
                self.someone_migrated += 1
                self.mig_binary = 1
                #migrant[0].migrated = True
                migrant[0].salary = util_migrate
                individual_set.loc[(individual_set.id == migrant[0].unique_id), 'ind'] = migrant[0]
                
        elif method == 'tpb': #Theory of Planned Behavior
            self.total_util_w_migrant = self.total_utility - migrant[0].salary + util_migrate
            # calculate TPB components
            #behavioral control
            if self.someone_migrated > 0:
                experience = 1
            else:
                experience = 0
                
            if self.network_moves > 0:
                network_exp = 1
            else:
                network_exp = 0
            
            if self.wealth == 0:
                mig_asset_percent = 0
            else:
                mig_asset_percent = (self.wealth - util_migrate)/self.wealth
            
            if mig_asset_percent < 0:
                mig_asset_percent = 0
            #y = 1/ (1+ e^(-5*x)) for 0 <= x <= 1 ######## play with values of coefficient 
            asset_rate = 1/(1 + np.exp(-self.k*mig_asset_percent)) #calibrate coefficient here
            self.control = self.weight1*asset_rate + self.weight2*experience + self.weight3*network_exp 
                             #test these weights
            #attitudes
            perceived_benefit = (util_migrate - migrant[0].salary) / util_migrate
            age_adj = migrant[0].age - 14
            if random.random() <= perceived_benefit:
                self.attitude = 0.028*(age_adj**2)*np.exp(-1*age_adj / 8)
            else:
                self.attitude = 0.014*(age_adj**2)*np.exp(-1*age_adj / 8)
            #network effect
            self.network_fact = (self.network_moves / self.network_size) + 1
            decision = tpb()
            decision.decide(self)
            if decision.outcome == True:
                self.wealth = self.wealth - mig_threshold #subtract out mig_threshold cost
                self.someone_migrated += 1
                self.mig_binary = 1
                #migrant[0].migrated = True
                migrant[0].salary = util_migrate
                individual_set.loc[(individual_set.id == migrant[0].unique_id), 'ind'] = migrant[0]
                
        elif method == 'pmt': #Protection Motivation Theory
            threat_threshold = self.threshold #this needs to be tuned 
            own_perception = self.num_shocked * random.random()
            comm_perception = community.num_impacted * random.random()
            severity = community.comm_impact * comm_perception
            if self.wealth > 0:
                vulnerability = ((self.land_owned * self.ag_factor) / self.wealth) * random.random()
            else:
                vulnerability = 0.9 
            threat = severity * vulnerability
            if threat >= threat_threshold: #if threshold is exceeded, calculate coping appraisal
                response_efficacy = self.network_moves / self.network_size
                if self.someone_migrated > 0:
                    self_efficacy = 1
                else:
                    self_efficacy = 0
                if self.wealth == 0:
                    mig_asset_percent = 0
                else:
                    mig_asset_percent = (self.wealth - util_migrate)/self.wealth
                if mig_asset_percent < 0:
                    mig_asset_percent = 0
                cost_efficacy = 1/(1 + np.exp(-self.k * mig_asset_percent))
                ## tune these weights 
                self.coping_appraisal = (self.weight1 * cost_efficacy) + (self.weight2 * self_efficacy) + (self.weight3 * response_efficacy)
            else:
                return
            decision = pmt()
            decision.decide(self)
            if decision.outcome == True:
                self.wealth = self.wealth - mig_threshold #subtract out mig_threshold cost
                self.someone_migrated += 1
                self.mig_binary = 1
                #migrant[0].migrated = True
                migrant[0].salary = util_migrate
                individual_set.loc[(individual_set.id == migrant[0].unique_id), 'ind'] = migrant[0]
                
        elif method == 'mobility_potential':
            land_ac  = self.land_owned / av_land
            wealth_ac = self.wealth / av_wealth 
            my_individuals = individual_set.loc[(individual_set['hh'] == self.unique_id, 'ind')]
            workers = 0
            for i in my_individuals:
                if i.employment != 'None':
                    workers += 1
            family_ac = workers / self.hh_size
            self.adaptive_capacity = land_ac + wealth_ac + family_ac
            self.mobility_potential = self.rootedness
            x = self.num_shocked
            if x == 0:
                if self.adaptive_capacity >= 3 and self.mobility_potential >= 0.5: #high AC, high MP
                    self.unique_mig_threshold = 0.75
                elif self.adaptive_capacity < 3 and self.mobility_potential >= 0.5: #low AC, high MP
                    self.unique_mig_threshold = 0.6
                elif self.adaptive_capacity >= 3 and self.mobility_potential < 0.5: #high AC, low MP
                    self.unique_mig_threshold = 0.1
                else: #low AC and low MP
                    self.unique_mig_threshold = 0
            else:
                if self.adaptive_capacity >= 3 and self.mobility_potential >= 0.5: #high AC, high MP
                    self.unique_mig_threshold = 300*((1/0.5) * (x**(6-1)) * (np.exp(-(1/0.5) * x))) / (6*5*4*3*2)
                elif self.adaptive_capacity < 3 and self.mobility_potential >= 0.5: #low AC, high MP
                    self.unique_mig_threshold = 5*((1/0.8) * (x**(2-1)) * (np.exp(-(1/0.8) * x))) / (2)
                elif self.adaptive_capacity >= 3 and self.mobility_potential < 0.5: #high AC, low MP
                    self.unique_mig_threshold = 1/(1 + np.exp(-0.75*(x-3)))
                else: #low AC and low MP
                    self.unique_mig_threshold = 1/(1 + np.exp(-2*(x-4)))
                        
            decision = mobility_potential()
            decision.decide(self)
            if decision.outcome == True:
                self.wealth = self.wealth - mig_threshold #subtract out mig_threshold cost
                self.someone_migrated += 1
                self.mig_binary = 1
                #migrant[0].migrated = True
                migrant[0].salary = util_migrate
                individual_set.loc[(individual_set.id == migrant[0].unique_id), 'ind'] = migrant[0]
        
        
        elif method == 'hybrid':
            #threat threshold comes from mobility potential 
            land_ac  = self.land_owned / 14
            wealth_ac = self.wealth / av_wealth
            my_individuals = individual_set.loc[(individual_set['hh'] == self.unique_id, 'ind')]
            workers = 0
            for i in my_individuals:
                if i is not None:
                    if i.employment != 'None':
                        workers += 1
            family_ac = workers / self.hh_size
            self.adaptive_capacity = land_ac + wealth_ac + family_ac
            self.mobility_potential = self.rootedness
            x = self.num_shocked
            if self.adaptive_capacity >= 3 and self.mobility_potential >= 0.5: #high AC, high MP
                self.unique_mig_threshold = 250*((1/0.5) * (x**(6-1)) * (np.exp(-(1/0.5) * x))) / (6*5*4*3*2)
            elif self.adaptive_capacity < 3 and self.mobility_potential >= 0.5: #low AC, high MP
                self.unique_mig_threshold = 5*((1/0.8) * (x**(2-1)) * (np.exp(-(1/0.8) * x))) / (2)
            elif self.adaptive_capacity >= 3 and self.mobility_potential < 0.5: #high AC, low MP
                self.unique_mig_threshold = 1/(1 + np.exp(-1*(x-3)))
            else: #low AC and low MP
                self.unique_mig_threshold = 1/(1 + np.exp(-4*(x-3)))
                        
            threat_threshold = self.unique_mig_threshold #this is coming from mobility potential math
            own_perception = self.num_shocked * random.random()
            comm_perception = community.num_impacted * random.random()
            severity = community.comm_impact * comm_perception
            if self.wealth > 0:
                vulnerability = ((self.land_owned * self.ag_factor) / self.wealth) * random.random()
            else:
                vulnerability = 0.9 
            threat = severity * vulnerability
            
            #threat appraisal first informed by PMT
            if threat >= threat_threshold: #if threshold is exceeded, calculate coping appraisal
                #this part coming from TPB
                self.total_util_w_migrant = self.total_utility - migrant[0].salary + util_migrate
                # calculate TPB components
                #behavioral control
                if self.someone_migrated > 0:
                    experience = 1
                else:
                    experience = 0
                if self.network_moves > 0:
                    network_exp = 1
                else:
                    network_exp = 0
                if self.wealth == 0:
                    mig_asset_percent = 0
                else:
                    mig_asset_percent = (self.wealth - util_migrate)/self.wealth
                if mig_asset_percent < 0:
                    mig_asset_percent = 0
                #y = 1/ (1+ e^(-5*x)) for 0 <= x <= 1 ######## play with values of coefficient 
                asset_rate = 1/(1 + np.exp(-self.k*mig_asset_percent)) #calibrate coefficient here
                self.control = self.weight1*asset_rate + self.weight2*experience + self.weight3*network_exp 
                #test these weights
                #attitudes
                perceived_benefit = (util_migrate - migrant[0].salary) / util_migrate
                age_adj = migrant[0].age - 14
                if random.random() <= perceived_benefit:
                    self.attitude = 0.028*(age_adj**2)*np.exp(-1*age_adj / 8)
                else:
                    self.attitude = 0.014*(age_adj**2)*np.exp(-1*age_adj / 8)
                #network effect
                self.network_fact = (self.network_moves / self.network_size) + 1
            else:
                return
            
            decision = hybrid()
            decision.decide(self)
            if decision.outcome == True:
                self.wealth = self.wealth - mig_threshold #subtract out mig_threshold cost
                self.someone_migrated += 1
                self.mig_binary = 1
                #migrant[0].migrated = True
                migrant[0].salary = 0#util_migrate
                individual_set.loc[(individual_set.id == migrant[0].unique_id), 'ind'] = migrant[0]
                self.mig_angent_id=migrant[0].unique_id
        else:
            return
        
        if decision.outcome == True:
            return True#,migrant[0].unique_id
            
            
            
        else:
            pass


    
    def sum_utility(self, individual_set):
        my_individuals = individual_set.loc[(individual_set['hh'] == self.unique_id, 'ind')]
        sum_util = 0
        for i in my_individuals:
            if i is not None:
                sum_util = sum_util + i.salary
        self.total_utility = sum_util

        if self.total_utility < self.wellbeing_threshold:
            self.secure = False
        else:
            self.secure = True 

    def hire_employees(self,working_month): #how many people to hire? and wtp 
        if self.land_impacted == False and working_month:
            self.num_employees = round(self.land_owned / 10) #initially 2 then 3
        else:
            self.num_employees = 0 

        if self.num_employees > 0: 
            self.wtp = ((self.ag_factor * self.land_owned) / (self.num_employees + 1))
            self.wta = (self.wellbeing_threshold / self.hh_size) * random.random() 
        else:
            self.wtp = 0
            self.wta = (self.wellbeing_threshold / self.hh_size) * random.random()


    def update_wealth(self, individual_set):
        #update wealth here
        my_individuals = individual_set.loc[(individual_set['hh'] == self.unique_id, 'ind')]
        sum_salaries = 0  
        #sum across all salaries 
        for i in my_individuals:
            if i is not None:
                sum_salaries = sum_salaries + i.salary
        
        self.wealth = self.wealth + sum_salaries - self.expenses - np.sum(self.payments) + self.land_prod
        
        if self.wealth < 0:
            self.wealth = 0 
            self.secure = False 

        #reset these values
        self.land_impacted = False
        self.land_prod = self.ag_factor * self.land_owned
        self.employees = []

    def set_network(self, hh_set, network):
        list_look = nx.to_dict_of_lists(network)
        self.hh_network = list_look[self.unique_id-1]

    def check_network(self, hh_set):
        self.network_moves = 0
        for a in self.hh_network:
            look = hh_set.loc[(hh_set.hh_id == (a+1)), 'household']
            if type(look[0])==pd.Series:
                self.network_moves += look[0].iloc[0].mig_binary
            else:
                self.network_moves += look[0].mig_binary
