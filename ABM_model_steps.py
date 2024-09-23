#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the actual model with scheduling and steps for
 ABM of environmental migration

@author: kelseabest
version edited by Orla O'Neill
"""

#import packages
from individual import *
from community import *
from decisions import *
from hh_class import *
from weather_check import *
from hh_class_for_mirgants import *
import random
import math
import numpy as np
import pandas as pd
import networkx as nx
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader

#initialize model
class ABM_Model:
    def __init__(self, decision, mig_util, mig_threshold, wealth_factor, ag_factor, comm_scale, shock_method, network_type, w1, w2, w3, k,
     threshold,senario,testing=False,factor=100000,binary=False):
        self.decision = decision #set decision type
        self.mig_util = mig_util #utility to migrate
        self.mig_threshold = mig_threshold #threshold to migrate
        self.senario=senario
        
        #check weather for the senario
        weather=check_weather(self.senario)
        if binary:
            self.F,self.C,self.H=weather.binary_checker()
            self.ft='true'
        else:
            self.F,self.C,self.H=weather.normalised_checker()
            self.ft='false'

        #setting time values
        self.time=self.F.time
        self.tick=0
        
        self.migrations = pd.DataFrame()#Initialize number of overall migrations
        self.wealth_factor = wealth_factor #scale of initial household wealth
        self.av_wealth = 0 #this is to store community level average wealth 
        self.av_land = 0 #this is to store community level average land
        self.ag_factor = ag_factor #scalar of relationship between land and wealth
        self.comm_scale = comm_scale #scale (% community) impacted by an environmental shock
        self.shock_method = shock_method #this can be "shock" or "slow_onset"
        
        self.network_type = network_type
        self.network_size = 3
        self.network = nx.Graph() #initialize empty network to be generated 
        
        #weights for TPB
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.k = k 
        
        #threshold for PMT
        self.threshold = threshold
        
        self.mig_total_total=0

        #load in census data
        if testing:
            #testing data set for trial runs
            self.df_census=pd.read_excel(r"Bangladesh cencus Data\2011\testing_upazilas.xlsx",skiprows=9)
            self.df_census.index=self.df_census.Code
            self.df_census=self.df_census.drop('Code',axis=1)
            self.df_census=self.df_census.drop('Unnamed: 0',axis=1)
            self.df_census=self.df_census.rename(columns={'Upazila/Thana Name':'Upazila'})
        else:
            self.df_census=pd.read_excel(r"CBangladesh cencus Data\census_2011_with_common_crops.xlsx")
            self.df_census.index=self.df_census.Code
            self.df_census=self.df_census.drop('Code',axis=1)
        
        #find the number of agents
        self.df_individual = self.df_census['Total'].apply(lambda x: math.ceil(x / factor))
        self.df_individual=self.df_individual[:-1]
        
        #find the number of households
        self.df_hh=(self.df_census['hh_number'].apply(lambda x: math.ceil(x / factor)))[:-1]
        self.step_time=self.time.isel(time=0)
        
        #read in upazila shapefile
        bgd_shapefile_path = r"shapefiles\gadm41_BGD_3.shp" 
        self.gdf = gpd.read_file(bgd_shapefile_path)

        #for storing data
        for ID in self.df_hh.index:
            data_set_key = f'data_set_{ID}'
            origin_comm_key=f'origin_comm_{ID}'
            ag_fac_key=f'ag_fac_{ID}'
            jobs_avail_key=f'jobs_avail_{ID}'

            self.__dict__[ag_fac_key]=self.ag_factor
            self.__dict__[jobs_avail_key] = math.ceil((self.df_census.loc[ID,'Industry']+self.df_census.loc[ID,'Service'])/factor) #number of non_ag jobs in community
            self.__dict__[data_set_key] = pd.DataFrame()
            self.__dict__[origin_comm_key] = origin(self.df_hh[ID], self.__dict__[jobs_avail_key], self.comm_scale,
                                                    self.F.sel(region=int(ID)),self.C.sel(region=int(ID)),self.H.sel(region=int(ID)),
                                                    self.__dict__[ag_fac_key])
            
            
        self.last = pd.DataFrame()
        self.mig_df=pd.DataFrame()
        
        #set up agents for each upazila
        for x in self.df_hh.index:
            self.set_up_agents(x)
            
            

    def set_up_agents(self, ID):
        # Create individuals
        individual_set_key = f'individual_set_{ID}'
        ind_key=f'ind_{ID}'
        ag_fac_key=f'ag_fac_{ID}'
        self.__dict__[individual_set_key] = pd.DataFrame()
        for i in range(self.df_individual.loc[ID]):
            ind_key = Individual(self.__dict__[ag_fac_key],ID)
            row = pd.DataFrame({'ind': [ind_key], 'id': [ind_key.unique_id],
                                'age': [ind_key.age], 'hh': [ind_key.hh],
                                'gender': [ind_key.gender],
                                'originally_from':[ind_key.originally_from],'currently_living':np.nan,
                                'mig_id':None,'mig_dest':None})
            self.__dict__[individual_set_key] = pd.concat([self.__dict__[individual_set_key], row])
    
        # Create households
        hh_set_key = f'hh_set_{ID}'
        a_key =f'a_{ID}'
        got_job_key = f'got_job_{ID}'
        self.__dict__[hh_set_key] = pd.DataFrame()  # Empty DataFrame to store agents created
        for i in range(self.df_hh.loc[ID]):
            self.__dict__[a_key] = Household(self.wealth_factor, self.ag_factor, self.w1, self.w2, self.w3, self.k, self.threshold,self.df_individual.loc[ID]/self.df_hh.loc[ID])
            self.__dict__[a_key].gather_members(self.__dict__[individual_set_key])
            self.__dict__[a_key].assign_head(self.__dict__[individual_set_key])
            row = pd.DataFrame({'household': [self.__dict__[a_key]], 'hh_id': [self.__dict__[a_key].unique_id],
                                'wtp': [self.__dict__[a_key].wtp], 'wta': [self.__dict__[a_key].wta],'type':['normal']})
            self.__dict__[hh_set_key] = pd.concat([self.__dict__[hh_set_key], row])
    
        
        self.average_land(ID)
        self.generate_network(ID)
        
        self.__dict__[got_job_key] = 0 #tracks successful job in labor market
        
        for i in self.__dict__[hh_set_key]['household']: #set network
            i.set_network(self.__dict__[hh_set_key], self.network)
        #setting the hh id to 1 for the next upazila
        Household.next_uid=1

    def model_step(self):
        self.step_time=self.time.isel(time=self.tick)

        #run through each upazila, probably could do this in random order in future
        for ID in (self.df_hh.index):
            individual_set_key = f'individual_set_{ID}'
            hh_set_key = f'hh_set_{ID}'
            ag_fac_key=f'ag_fac_{ID}'
            origin_comm_key=f'origin_comm_{ID}'
            
            #random schedule each time
            random_sched_hh = np.random.permutation((self.__dict__[hh_set_key].hh_id))
            random_sched_ind = np.random.permutation((self.__dict__[individual_set_key].id))

            #change agricultural productivity based on the weather
            if self.shock_method=='weather': 
                self.__dict__[origin_comm_key].shock(self.step_time)
                self.__dict__[ag_fac_key] = self.__dict__[origin_comm_key].ag_factor
            else: 
                self.__dict__[ag_fac_key] = self.__dict__[ag_fac_key] * 0.95 #5% decrease in productivitiy each step 
            
            self.average_wealth(ID)
                #households need to check land
            for i in random_sched_hh: #these are the steps at each tick for hh
                agent_var = self.__dict__[hh_set_key][self.__dict__[hh_set_key].hh_id == i].household
                agent_var_0=agent_var[0]
                
                agent_var_0.check_land(self.__dict__[origin_comm_key], self.comm_scale)
                
                #find if the dominant crop is in season and if so, hire employees
                step_time=self.time.isel(time=self.tick).values
                month = step_time.astype('datetime64[M]').astype('int').item() % 12 + 1
                start_month=self.df_census.loc[ID,'harvest_start']
                end_month=self.df_census.loc[ID,'reset_month']
                #tf= True or False
                if start_month <= end_month:
                    tf=start_month <= month <= end_month
                else:
                    tf=month >= start_month or month <= end_month
                agent_var_0.hire_employees(tf) 
    
                #individuals look for work
            for j in random_sched_ind: #steps for individuals

                ind_var = self.__dict__[individual_set_key][self.__dict__[individual_set_key].id == j].ind
                if ind_var[0] is not None:
                    ind_var[0].check_eligibility()
                    ind_var[0].find_work(self.__dict__[hh_set_key], self.mig_util)
                else:
                    pass
    
            #double auction at model level 
            self.double_auction(ID)
                    
            if self.tick>0:
                    #households decide to send a migrant or not and update wealth
                for i in random_sched_hh: #these are the steps at each tick for hh
                    hh_var = self.__dict__[hh_set_key][self.__dict__[hh_set_key].hh_id == i].household
                    if hh_var[0] is not None:
                        hh_var[0].check_network(self.__dict__[hh_set_key])
                        hh_var[0].sum_utility(self.__dict__[individual_set_key])
                        outcome=hh_var[0].migrate(self.decision, self.__dict__[individual_set_key],self.__dict__[hh_set_key], self.mig_util, self.mig_threshold, self.__dict__[origin_comm_key], self.av_wealth, self.av_land)
                        
                        hh_var[0].update_wealth(self.__dict__[individual_set_key])
                        
                        if outcome:
                            #if it is decided that a migrant will be sent, choose a destination and move them
                            mig_agent_id=hh_var[0].mig_angent_id
                            best_location=self.pull_calculation(ID,mig_agent_id) 
                            self.move_agent(ID,mig_agent_id,best_location)


    
    def pull_calculation(self,agent_ID,mig_agent_id): 
        pull=[] 
        individual_set_key = f'individual_set_{agent_ID}'

        #set up for return migration, but ran out of time to actually make it work
        if self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == mig_agent_id].ind.values[0].migrated==True:
            print('!!!')
            if self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == mig_agent_id,'originally_from'].values[0]==self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == mig_agent_id,'currently_living'].values[0]:
                print('migrant going back')
                return self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == mig_agent_id,'mig_dest'].values[0]
                
            else:
                print('migrant going home')
                return self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == mig_agent_id,'originally_from'].values[0]
                
        else:
            for ID in self.df_hh.index: #want to change to within radius of affordability 
                data_set_key = f'data_set_{ID}' 
                hh_set_key = f'hh_set_{ID}'  
                if ID==agent_ID: 
                    #make sure the agent doesn't choose original upazila
                    pull.append(-999) 
                else:
                    #find number of agri jobs avalible
                    no_jobs=0
                    for k in self.__dict__[hh_set_key].household.values:
                        no_jobs+=k.num_employees
                    #find average of these payments
                    money_to_be_made=self.__dict__[data_set_key].wtp.values[-len(self.__dict__[hh_set_key]):]
                    #calculate pull
                    pull.append(np.mean(money_to_be_made)*no_jobs) 

            #find destination with the highest pull
            max_pull=max(pull) 
            max_pull_loc=pull.index(max_pull) 
            max_pull_ID=self.df_hh.index[max_pull_loc]
        
            return max_pull_ID
    
    def move_agent(self,agentid,j,ID):
        #create a new hh in that upazila
        #add the migrant to that hh
        hh_set_key = f'hh_set_{ID}'
        hh_old_set_key=f'hh_set_{agentid}'
        a_key =f'a_{agentid}'
        b_key=f'b_{ID}'
        individual_set_key = f'individual_set_{agentid}'
        individual_mig_set_key=f'individual_set_{ID}'
        data_set_key=f'data_set_{ID}'

        for i in self.__dict__[hh_set_key].household:
            i.mig_arr+=1

        #set up for return migration, but ran out of time to actually make it work
        if self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == j].ind[0].migrated==True:
            old_hh_id=self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id==j,'ind'][0].hh
            self.__dict__[individual_mig_set_key].loc[self.__dict__[individual_mig_set_key].id==j,'ind']=self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id==j,'ind']

            mask = self.__dict__[data_set_key]['agent IDs'].apply(lambda x: np.isin(j, x))
            hh_no=self.__dict__[data_set_key][mask].hh_id[0].values[0]
            self.__dict__[individual_mig_set_key].loc[self.__dict__[individual_mig_set_key].id==j,'hh']=hh_no
            self.__dict__[individual_mig_set_key].loc[self.__dict__[individual_mig_set_key].id==j,'ind'][0].hh=hh_no
            self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id==j,'hh']=hh_no
            self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id==j,'ind'][0].hh=hh_no

            self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == j,'currently_living']= int(ID)
            
            self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id==j,'ind']=None
            
            
        else:
            ##what is in the origional hh creation
            self.__dict__[b_key] = Migrant(self.wealth_factor, self.ag_factor, self.w1, self.w2, self.w3, self.k, self.threshold)
            self.__dict__[b_key].hh_network=self.__dict__[a_key].hh_network
            self.__dict__[b_key].unique_id=(max(self.__dict__[hh_set_key].hh_id.values)+1)
            
            ##creation of a new hh for migrant
            self.__dict__[b_key].gather_members(self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == j])
            self.__dict__[b_key].assign_head(self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == j])
            row = pd.DataFrame({'household': [self.__dict__[b_key]], 'hh_id': [self.__dict__[b_key].unique_id],
                                'wtp': 0, 'wta': [self.__dict__[b_key].wta],'type':['migrant']})
            self.__dict__[hh_set_key] = pd.concat([self.__dict__[hh_set_key], row])
            
            #changing features of agent in origional upazila
            self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == j,'currently_living']= int(ID)
            self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == j,'mig_id']=self.__dict__[b_key].unique_id
            
            #adding migrant to new upazila
            self.__dict__[individual_mig_set_key]=pd.concat([self.__dict__[individual_mig_set_key],self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == j]])
            
            #changing hh in new upazila
            self.__dict__[individual_mig_set_key].loc[self.__dict__[individual_mig_set_key].id == j,'hh'] = self.__dict__[b_key].unique_id
            self.__dict__[individual_mig_set_key].loc[self.__dict__[individual_mig_set_key].id==j,'ind'][0].hh=self.__dict__[b_key].unique_id
            
            self.__dict__[individual_mig_set_key].loc[self.__dict__[individual_mig_set_key].id == j,'mig_dest']=int(ID)
            self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == j,'mig_dest']=int(ID)

            #removing the agent from origional upazila
            self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id == j].ind.migrated=True
            self.__dict__[individual_mig_set_key].loc[self.__dict__[individual_mig_set_key].id == j].ind.migrated=True
            self.__dict__[individual_set_key].loc[self.__dict__[individual_set_key].id==j,'ind']=None
            self.__dict__[individual_mig_set_key].loc[self.__dict__[individual_mig_set_key].id == j,'employment']='None'
            

    def double_auction(self,ID): #gets people looking for work and hh employing
        individual_set_key = f'individual_set_{ID}'
        hh_set_key = f'hh_set_{ID}'
        got_job_key = f'got_job_{ID}'
        origin_comm_key=f'origin_comm_{ID}'
        poss_employees = []  
        poss_employers = [] 
        still_looking_skilled = []
        still_looking_unskilled = []
        auctions = 3 # rounds w/ nothing changing 
        static_rounds = 0 

        for i in self.__dict__[individual_set_key]['ind']:
            if i is not None:
                if i.employment == "Looking":
                    poss_employees.append(i)
            # else:
            #     pass
        if poss_employees == None:
            return
        for h in self.__dict__[hh_set_key]['household']:
            if h.num_employees > 0:
                poss_employers.append(h)
        if poss_employers == None:
            return 

        all_looking = len(poss_employees)
        
        while static_rounds < auctions and all_looking > 0: 
            changed = False 
            for a in poss_employers: #households pick some people
                if a.num_employees > 0: 
                    if a.num_employees > len(poss_employees):
                        random_inds_look =  np.random.choice(poss_employees, len(poss_employees))
                    else:
                        random_inds_look =  np.random.choice(poss_employees, a.num_employees)
                    
                    for random_ind in random_inds_look:
                        if random_ind.employment != "Looking":
                            pass 
                        elif a.wtp >= random_ind.wta:
                            a.employees.append(a)
                            a.num_employees = a.num_employees - 1
                            random_ind.salary = (random_ind.wta + a.wtp)/2
                            random_ind.employment = "OtherAg"
                            changed = True 
                            random_ind.employer = a.unique_id
                            a.payments.append(random_ind.salary)
                            all_looking = all_looking - 1 
                            self.__dict__[got_job_key] += 1 
                            self.__dict__[individual_set_key].loc[(self.__dict__[individual_set_key].id == random_ind.unique_id), 'ind'] = random_ind
                self.__dict__[hh_set_key].loc[(self.__dict__[hh_set_key].hh_id == a.unique_id), 'household'] = a
                
            if changed:
                static_rounds = 0 
            else:
                static_rounds += 1 
                
                

        #individuals may look for an unskilled or a skilled job within the community 
        for i in self.__dict__[individual_set_key]['ind']:
            if i is not None:
                if i.employment == "Looking":
                    my_hh = self.__dict__[hh_set_key][self.__dict__[hh_set_key]['hh_id'] == i.hh]['household']
                    if type(my_hh[0])==pd.Series:
                        my_hh_0=my_hh[0].iloc[0]
                    else:
                        my_hh_0=my_hh[0]
                    if my_hh_0.wealth > self.wealth_factor:
                        still_looking_skilled.append(i)
                    else:
                        still_looking_unskilled.append(i)
        if still_looking_unskilled == None and still_looking_skilled == None:
            return 
        
        if len(still_looking_unskilled) > self.__dict__[origin_comm_key].avail_jobs / 2:
            found_other_job_unskilled = random.sample(still_looking_unskilled, round(self.__dict__[origin_comm_key].avail_jobs / 2))
        else:
            found_other_job_unskilled = still_looking_unskilled

        if len(still_looking_skilled) > self.__dict__[origin_comm_key].avail_jobs / 2:
            found_other_job_skilled = random.sample(still_looking_skilled, round(self.__dict__[origin_comm_key].avail_jobs / 2))
        else:
            found_other_job_skilled = still_looking_skilled

        for i in found_other_job_unskilled:
            i.employment = "OtherNonAg_Unskilled"
            i.salary = 24000 * random.random() #some small number
            self.__dict__[individual_set_key].loc[(self.__dict__[individual_set_key].id == i.unique_id), 'ind'] = i

        for i in found_other_job_skilled:
            i.employment = "OtherNonAg_Skilled"
            i.salary = 50000 * random.random() #some greater number
            self.__dict__[individual_set_key].loc[(self.__dict__[individual_set_key].id == i.unique_id), 'ind'] = i

    def generate_network(self,ID): #create community level network with networkx 
        
        if self.network_type == 'random':
            self.network = nx.fast_gnp_random_graph(self.df_hh.loc[ID], 0.1) 
        if self.network_type == 'none':
            pass
        if self.network_type == 'small_world':
            self.network = nx.watts_strogatz_graph(self.df_hh.loc[ID], self.network_size, 0.15)
        if self.network_type == 'preferential':
            self.network = nx.barabasi_albert_graph(self.df_hh.loc[ID], self.network_size)
        if self.network_type == 'fully_connected':
            self.network = nx.complete_graph(self.df_hh.loc[ID])
    
    def average_wealth(self,ID):
        hh_set_key = f'hh_set_{ID}'
        self.av_wealth = 0
        for h in self.__dict__[hh_set_key]['household']:
            self.av_wealth += h.wealth
        self.av_wealth = self.av_wealth / self.df_hh.loc[ID]
        if self.av_wealth == 0: #this is to prevent 0 divisions
            self.av_wealth = 1
        
    def average_land(self,ID): 
        hh_set_key = f'hh_set_{ID}'
        self.av_land = 0
        for h in self.__dict__[hh_set_key]['household']:
            self.av_land += h.land_owned
        self.av_land = self.av_land / self.df_hh.loc[ID]
        
    def data_collect(self): #use this to collect model level data
    #household level data
        self.mig_total =0
        
        for ID in self.df_hh.index:
            counter=0
            individual_set_key = f'individual_set_{ID}'
            hh_set_key = f'hh_set_{ID}'
            data_set_key = f'data_set_{ID}'
            ag_fac_key=f'ag_fac_{ID}'
            got_job_key = f'got_job_{ID}'
            jobs_avail_key=f'jobs_avail_{ID}'
            
            step_time=self.time.isel(time=self.tick).values
            month = step_time.astype('datetime64[M]').astype('int').item() % 12 + 1
            year=np.datetime64(step_time, 'Y').astype(int)
            
            for j in self.__dict__[hh_set_key].hh_id.values:
                hh_var = self.__dict__[hh_set_key][self.__dict__[hh_set_key].hh_id == j].household
                hh = hh_var[0]
                emp=[]
                dest=[]
                for i in (self.__dict__[individual_set_key][self.__dict__[individual_set_key].hh==j].ind.values):
                    
                    if i is not None:
                        emp.append(i.employment)
                        dest.append(i.mig_dest)
                    else:
                        emp.append('Migrated')
                        dest.append('Not sure')
                row = pd.DataFrame({'tick': [self.tick],'month':[month],'year':[year+1970],'hh_id': [hh.unique_id],'type':[self.__dict__[hh_set_key][self.__dict__[hh_set_key].hh_id == j].type.values][0], 'migrations': [hh.someone_migrated],
                                    'arrivals':[hh.mig_arr],'wealth': [hh.wealth], 'num_shocked':[hh.num_shocked], 
                                    'wtp': [hh.wtp], 'wta': [hh.wta],
                                    'found_work': [self.__dict__[got_job_key]],
                                    'employment type':[emp],
                                    'ag_fac': [self.__dict__[ag_fac_key]],
                                    #'mig_util':[self.mig_util], 'mig_threshold':[self.mig_threshold],
                                    #'comm_scale':[self.comm_scale],
                                    #'avalible_jobs':[self.__dict__[jobs_avail_key]],
                                    #'avalible jobs in origin':[self.origin_comm.avail_jobs],
                                    #'w1':[self.w1],
                                    #'w2':[self.w2],
                                    #'w3':[self.w3],
                                    #'k':[self.k],
                                    #'threshold':[self.threshold],
                                    'number_of_hh_members':[len(self.__dict__[individual_set_key][self.__dict__[individual_set_key].hh==j])],
                                    'agent IDs':[self.__dict__[individual_set_key][self.__dict__[individual_set_key].hh==j].id.values],
                                    #'ind_id':[self.__dict__[individual_set_key][self.__dict__[individual_set_key].hh==j].id],
                                    'from':[self.__dict__[individual_set_key][self.__dict__[individual_set_key].hh==j].originally_from.values],
                                    'mig_dest':[self.__dict__[individual_set_key][self.__dict__[individual_set_key].hh==j].mig_dest.values],
                                    'living in':[self.__dict__[individual_set_key][self.__dict__[individual_set_key].hh==j].currently_living.values]})
                self.__dict__[data_set_key] = pd.concat([self.__dict__[data_set_key], row])
                
                
            self.last = self.__dict__[data_set_key][self.__dict__[data_set_key]['tick'] == self.tick]
            #self.last_before_that = self.__dict__[data_set_key][self.__dict__[data_set_key]['tick'] == self.tick-1]
           
            self.mig_sum = self.last.loc[:,'migrations'].sum(axis=0)
            #print('no migrants:',self.mig_sum)
            #self.mig_last_time=self.last_before_that.loc[:,'migrations'].sum(axis=0)
            #print(self.mig_last_time)
            #row = pd.DataFrame({'tick': [self.tick], 'total_mig': [mig_sum]})
            #self.migrations = pd.concat([self.migrations, row])
            self.mig_total+=self.mig_sum#-self.mig_last_time
        if self.tick>0:
            mig_this_tick=self.mig_total-self.mig_df.iloc[-1].loc['total_mirgrants']
        else:
            mig_this_tick=0
        row=pd.DataFrame({'tick': [self.tick],'date':[self.time.isel(time=self.tick).values],
                          'migrants':[mig_this_tick],'total_mirgrants':[self.mig_total]})
        self.mig_df=pd.concat([self.mig_df,row])
        #self.mig_total_total+=self.mig_total
            

    #tick up model 
    def tick_up(self):
        self.tick += 1
        step_time=self.time.isel(time=self.tick).values
        month = step_time.astype('datetime64[M]').astype('int').item() % 12 + 1
        #is_january = month == 1
        for ID in self.df_hh.index:
            individual_set_key = f'individual_set_{ID}'
            #ag_fac_key=f'ag_fac_{ID}'
            origin_comm_key=f'origin_comm_{ID}'
            jobs_avail_key=f'jobs_avail_{ID}'
            #tick and reset key values
            
            self.__dict__[origin_comm_key].impacted = False
            #print('before:',self.__dict__[origin_comm_key].avail_jobs)
            self.__dict__[origin_comm_key].avail_jobs = self.__dict__[jobs_avail_key]
            if not ('origin:',self.__dict__[origin_comm_key].avail_jobs=='dictionary:',self.__dict__[jobs_avail_key]):
                print('Not matching')
            
            for j in self.__dict__[individual_set_key].id:
                ind_var = self.__dict__[individual_set_key][self.__dict__[individual_set_key].id == j].ind
                ind_var.employment='None'

                #if it is jan, everyone ages up
                if month == 1:
                    if ind_var[0] is not None:
                        if type(ind_var[0])==pd.Series:
                            for i in ind_var[0]:
                                if i is not None:
                                    i.age_up()
                        else:
                            ind_var[0].age_up()
                
            #if the dominant crop is no longer in season, reset the agricultural productivity
            if month==self.df_census.loc[ID,'reset_month']:
                self.__dict__[origin_comm_key].ag_factor=100
                
    def save_files(self,i=None):
        #function to save data
        #i is the model run if there is multiple runs
        x=self.senario
        y=self.decision
        z=self.network_type
        j=self.comm_scale

        # Create an Excel writer object
        if i==None:
            excel_writer = pd.ExcelWriter("model_senario_"+str(x)+"_method_"+str(y)+"_network_type_"+str(z)+'_binary_'+(self.ft)+".xlsx", engine='xlsxwriter')
        else:
            excel_writer = pd.ExcelWriter(str(i)+"_model_senario_"+str(x)+"_method_"+str(y)+"_network_type_"+str(z)+'_binary_'+(self.ft)+".xlsx", engine='xlsxwriter')
        for ID in self.df_hh.index:
            data_set_key = f'data_set_{ID}'
            # Write each DataFrame to a different sheet named by ID
            self.__dict__[data_set_key].to_excel(excel_writer, sheet_name=str(ID))
            #print(self.__dict__[data_set_key]['avalible_jobs'])
            

        # Save the Excel file
        excel_writer.close()
        if i==None:
            self.mig_df.to_excel('number_migrants_senario_'+str(x)+"_method_"+str(y)+"_network_type_"+str(z)+'_binary_'+(self.ft)+".xlsx")
        else:
            self.mig_df.to_excel(str(i)+'_number_migrants_senario_'+str(x)+"_method_"+str(y)+"_network_type_"+str(z)+'_binary_'+(self.ft)+".xlsx")
        

    def plotting(self):
        #plotting function for checking, not really used
        migrations=[]
        arrivals=[] 
        for ID in self.df_census.index[:-1]:
            data_key=f'data_set_{ID}'
            migrations.append(self.__dict__[data_key].iloc[-1].loc['migrations'])
            arrivals.append(self.__dict__[data_key].iloc[-1].loc['arrivals'])

        bgd_shapefile_path = r"\shapefiles\gadm41_BGD_3.shp" 
        gdf = gpd.read_file(bgd_shapefile_path)
        bgd_feature = ShapelyFeature(Reader(bgd_shapefile_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none',linestyle=':')

        gdf=gdf.sort_values(by='CC_3')

        cmap = plt.cm.RdYlGn_r
        min_value = np.min(migrations)
        max_value = np.max(migrations)
        norm = Normalize(vmin=min_value, vmax=max_value)

        fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

        ax.coastlines(resolution='50m')
        ax.add_feature(cfeature.LAND)
        ax.add_feature(bgd_feature)

        min_lon = np.inf
        max_lon = -np.inf
        min_lat = np.inf
        max_lat = -np.inf

        for index, row in gdf.iterrows():
            region_id = int(row['CC_3'])
            if region_id in self.df_census.index[:-1]:# and 600000>region_id>=550000:
                data_key=f'data_set_{region_id}'
                color = cmap(norm(self.__dict__[data_key].iloc[-1].loc['migrations']))
                ax.add_geometries([row['geometry']], ccrs.PlateCarree(), facecolor=color, edgecolor='black')

                min_lon = min(min_lon, row['geometry'].bounds[0])
                max_lon = max(max_lon, row['geometry'].bounds[2])
                min_lat = min(min_lat, row['geometry'].bounds[1])
                max_lat = max(max_lat, row['geometry'].bounds[3])

                ax.set_extent([min_lon, max_lon, min_lat, max_lat])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) 
        plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
        plt.title('Number of migrants leaving per region')
        plt.show()

        min_lon = np.inf
        max_lon = -np.inf
        min_lat = np.inf
        max_lat = -np.inf

        for index, row in gdf.iterrows():
            region_id = int(row['CC_3'])
            if region_id in self.df_census.index[:-1]:# and 600000>region_id>=550000:
                data_key=f'data_set_{region_id}'
                color = cmap(norm(self.__dict__[data_key].iloc[-1].loc['arrivals']))
                ax.add_geometries([row['geometry']], ccrs.PlateCarree(), facecolor=color, edgecolor='black')

                min_lon = min(min_lon, row['geometry'].bounds[0])
                max_lon = max(max_lon, row['geometry'].bounds[2])
                min_lat = min(min_lat, row['geometry'].bounds[1])
                max_lat = max(max_lat, row['geometry'].bounds[3])

                ax.set_extent([min_lon, max_lon, min_lat, max_lat])

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) 
        plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.05)
        plt.title('Number of migrants arriving per region')
        plt.show()