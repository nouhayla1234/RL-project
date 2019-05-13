import gym
from gym import error, spaces, utils
from gym.utils import seeding
import retrieve_plotv6 as ret
import random
import numpy as np
from functools import reduce
import csv
import xml.etree.ElementTree as ET

class FooEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        
        self.file_original = "anytown_master.spr"
        self.file_action = "new_anytown_master.spr" 

        self.datapath = "data_demand.csv"
        self.csvData = []
        self.active_row = 0
        self.retrieve_csvdata()

        self.nb_tests = len(self.csvData)

        self.pumpIds = ["PUMP71", "PUMP220", "PUMP229"]
        self.nb_nodes = 29
        self.sr_71 = 1
        self.sr_220 = 1
        self.sr_229 = 1
        self.sr_var = 0.05
        self.sr_min = 0.3
        self.sr_max = 1.2
        self.counter = 0
        self.max_episodess = 40
        self.end_episode = False
        self.action_space = spaces.Discrete(6) #0=do nothing, 1 = decrease71 and 2 = increase71 etc...
        low = np.array([0.0 for i in range(self.nb_nodes)])
        high = np.array([np.inf for i in range(self.nb_nodes)])
        self.observation_space = spaces.Box(0, np.inf, shape=(self.nb_nodes,), dtype=np.float32)

    
    def step(self, action):
       self.take_action(action)
       ob = self.get_state()
       reward = self.reward_()
       if self.counter > self.max_episodess: 
            self.end_episode = True
       return ob, reward, self.end_episode, {}


    def reset(self):
        self.counter = 0
        self.sr_71 = 1
        self.sr_220 = 1
        self.sr_229 = 1
        self.end_episode = False
        ret.randomize_demand(self.file_action)
        return(self.get_state())


    def take_action(self, action):

        pumpId = ""
        
        '''Choose the action'''
        if action == 1 and self.sr_71 > self.sr_min:
            pumpId = "PUMP71"
            self.sr_71 = self.sr_71 - self.sr_var
        elif action == 2 and self.sr_71 < self.sr_max:
            self.sr_71 = self.sr_71 + self.sr_var
            pumpId = "PUMP71"
        elif action == 3 and self.sr_220 > self.sr_min:
            self.sr_220 = self.sr_220 - self.sr_var
            pumpId = "PUMP220"
        elif action == 4 and self.sr_220 < self.sr_max:
            self.sr_220 = self.sr_220 + self.sr_var
            pumpId = "PUMP220"
        elif action == 5 and self.sr_229 > self.sr_min:
            self.sr_229 = self.sr_229 - self.sr_var
            pumpId = "PUMP229"
        elif action == 6 and self.sr_229 < self.sr_max:
            self.sr_229 = self.sr_229 + self.sr_var
            pumpId = "PUMP229"

        if pumpId != "":
            speed_ratio = [self.sr_71, self.sr_220, self.sr_229]
            current_dictpump = ret.read_performance_curves(self.file_action)
            dict_ = ret.apply_affinity_laws(self.pumpIds, current_dictpump, speed_ratio)
            ret.write_performance_curve(pumpId, self.file_action, self.file_action, dict_)
            ret.launch_staci(self.file_action)
        self.counter += 1 


    def reward_(self):
        dict_pumps = ret.read_performance_curves(self.file_action)
        P_functions = [ret.polyfit_curves(element, dict_pumps) for element in self.pumpIds]
        Efficiencies = []
        counter2 = 0
        for element in self.pumpIds:
            Q_star = dict_pumps[element]['volume_flow_rate']
            P_star = P_functions[counter2](Q_star)
            H_star = dict_pumps[element]['headloss']
            efficiency = ret.get_pump_efficiencies(Q_star, P_star, H_star)
            if efficiency >= 1 and efficiency < 0:
                efficiency = 0
            Efficiencies.append(efficiency)
            counter2 +=1

        return reduce(lambda x, y: x*y,Efficiencies)


    def get_state(self):
        """Get the observation."""
        ob = np.array(ret.read_nodes_pressure(self.file_action), dtype=np.float32)
        return ob


    def seed(self, seed):
        random.seed(seed)
        np.random.seed


    def retrieve_csvdata(self):
        with open(self.datapath, 'r') as readFile:
            reader = csv.reader(readFile)
            self.csvData = list(reader)
            #print(self.csvData)
        readFile.close()


    def randomize_demand_csv(self, active_row): # We change the demand of the original_file
        tree = ET.parse(self.file_action)
        root = tree.getroot()
        counter = 2 # 0:reward, 1:speed ratio, 2~31:random demand

        for child in root:
            for node in child.findall('node'):
                demand = node.find('demand')
                new_demand = self.csvData[active_row][counter]
                demand.text = new_demand
                counter += 1

        tree.write(self.file_action, encoding='utf8')


