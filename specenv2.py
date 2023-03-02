import gym
from gym import spaces, utils, logger
import numpy as np
import random
import socket
from typing import Optional, List, Dict, Union
from messenger import Messenger
import time
import logging
import torch
import matplotlib.pyplot as plt
logging.basicConfig(level=logging.INFO, format='%(message)s')

class SpectrumEnv:
    def __init__(self, port=int(7500)):
      self._identifier = 'p1'
      self._server_address: str = '127.0.0.1'
      self._server_port: int = port
      self._socket: socket = None
      self._messenger: Optional[Messenger] = None
      self._sta_list: Optional[List] = None
      self._freq_channel_list: Optional[List] = None
      self._num_unit_packet_list: Optional[List] = None
      self._first_step: bool = True
      self._state: str = 'unconnected'
      self._score: Dict = {'packet_success': 0, 'packet_delayed': 0, 'packet_dropped': 0,
                             'collision': 0, 'total_score': 0}
      self._freq_channel_list = []
      self._sta_list = []
      self._num_unit_packet_list = []
      self._num_freq_channel = 0
      self._num_sta = 0
      self.sensed_list = []
      
      self.packet_success = 0
      self.packet_delayed = 0
      self.packet_dropped = 0
      self.collision = 0
      self.total_score = 0
      self.num_unit_packet = 0
      self.reward = self.packet_success - 4 * self.collision
    
      # action_space
      #self.action_space = spaces.Discrete(16)
    
      # observation_space
      #self.obs = spaces.Box(low = 0, high = 1, shape =(1e6,4), dtype = int).sample()
      self.obs = np.zeros((1200000,4),dtype = int)
      self.yes_time = []
      self.yes = []
      self.no_time =[]
      self.no = []
      self.oh_time =[]
      self.oh = []
      self.count = 512
      self.sta_allocation_dict = {}

      self.observation_space = self.obs[self.count-512 : self.count]
      self.check = 0
      self.action_space = 5
    def reset(self):
      self.packet_success = 0
      self.packet_delayed = 0
      self.packet_dropped = 0
      self.collision = 0
      self.total_score = 0
      self.reward = 0
      self.num_unit_packet = 0
      self.obs = np.zeros((1200000,4),dtype = int)
      self.yes_time = []    # yes is success
      self.yes = []
      self.no_time =[]      # no is fail
      self.no = []
      self.oh_time =[]      # oh is empty
      self.oh = []
      self.count = 512
      self.sensed_list = []
      self.sta_allocation_dict = {'0':-1,'1':-1,'2':-1,'3':-1}
      self.check = 0
      #self._first_step = True

    def step(self, action):
        #print('step')
        if self._first_step:
            #print('step2')
            self._messenger.recv()  # Discard the first observation
            self._first_step = False
        #print('step3')
        #self.check = 3
        # action == 0 <-> sensing
        if (action[0]<=0) and (action[1]<=0) and (action[2]<=0) and (action[3]<=0):
            self.check = 4
            #print('step4')
            self.sensed_list = []
            self._messenger.send('action', {'type':'sensing'})
            msg_type, msg = self._messenger.recv()
            
            if msg_type == 'observation':
                self.sensed_list = msg['observation']['sensed_freq_channel']
            
            for i in range(4):
                if i in self.sensed_list:
                    self.obs[self.count][i] = -1
                    self.no.append(i)
                    self.no_time.append(self.count)
                else:
                    self.obs[self.count][i] = 0 #empty
                    self.oh.append(i)
                    self.oh_time.append(self.count)
                i += 1
                is_end = False
            self.count += 1
            obs = self.obs[self.count-512 : self.count].flatten()
            reward = -0.1
            if msg_type == 'simulation_finished':
                is_end = True
            return obs, reward, is_end
        ##########################################
        # tx_data_packet
        else:
            is_end = False
            #print('step5')
            #self.check = 5
            sta_list = []
            sta_allocation_dict = {'0':-1,'1':-1,'2':-1,'3':-1}
            if action[0] >= 0:
                sta_allocation_dict['0'] = 0
            else:
                sta_allocation_dict['0'] = -1
            if action[1] >= 0:
                sta_allocation_dict['1'] = 1
            else:
                sta_allocation_dict['1'] = -1            
            if action[2] >= 0:
                sta_allocation_dict['2'] = 2
            else:
                sta_allocation_dict['2'] = -1
            if action[3] >= 0:
                sta_allocation_dict['3'] = 3
            else:
                sta_allocation_dict['3'] = -1
            sta_list.append(sta_allocation_dict['0'])
            sta_list.append(sta_allocation_dict['1'])
            sta_list.append(sta_allocation_dict['2'])
            sta_list.append(sta_allocation_dict['3'])
        
            if action[4] >= 0:
                num_unit_packet = 2
            else:
                num_unit_packet = 1        
            #print('step6')
            #print('action', {'type': 'tx_data_packet', 'sta_allocation_dict': sta_allocation_dict,
            #                  'num_unit_packet': num_unit_packet})
            self._messenger.send('action', {'type': 'tx_data_packet', 'sta_allocation_dict': sta_allocation_dict,
                              'num_unit_packet': num_unit_packet})
            msg_type , msg = self._messenger.recv()
            #print('step7')
            if msg:
                success = msg['observation']['success_freq_channel']
                if msg_type == 'observation':
                    for i in sta_list:
                        if i in success:
                            for j in range(24*num_unit_packet):
                                self.obs[self.count+j][i] = 1
                                self.yes.append(i)
                                self.yes_time.append(self.count+j)
                        else:
                            for j in range(24*num_unit_packet):
                                self.obs[self.count+j][i] = -1
                                self.no.append(i)
                                self.no_time.append(self.count+j)
                self.count += 24*num_unit_packet
                self.packet_success = msg['score']['packet_success'] - self._score['packet_success']
                self.packet_delayed = msg['score']['packet_delayed'] - self._score['packet_delayed']
                self.packet_dropped = msg['score']['packet_dropped'] - self._score['packet_dropped']
                self.collision = msg['score']['collision'] - self._score['collision']
                self._score['packet_success'] = msg['score']['packet_success']
                self._score['packet_delayed'] = msg['score']['packet_delayed']
                self._score['packet_dropped'] = msg['score']['packet_dropped']
                self._score['collision'] = msg['score']['collision']
                self._score['total_score'] = self._score['packet_success'] - 4*self._score['collision']
                is_end = False
                #reward = msg['reward']
                reward = self.packet_success - 2*self.collision
                if msg['score']:
                    self.total_score = msg['score']['total_score']
            else:
                reward = -0.01
        
            obs = self.obs[self.count-512 : self.count].flatten()
            if msg_type == 'simulation_finished':
                is_end = True
            return obs, reward, is_end

         


    def start_simulation(self, time_us):
        if not ((isinstance(time_us, float) or isinstance(time_us, int)) and time_us > 0):
            logging.warning('Simulation time should be a positive number.')
            return
        if self._state == 'idle':
            self._messenger.send('start_simulation', time_us)
            msg_type, msg = self._messenger.recv()
            if msg_type == 'operator_info':
                self._sta_list = msg['sta_list']
                self._freq_channel_list = msg['freq_channel_list']
                self._num_unit_packet_list = msg['num_unit_packet_list']
                self._first_step = True
                self._score = {'packet_success': 0, 'packet_delayed': 0, 'packet_dropped': 0,
                               'collision': 0, 'total_score': 0}
                self._state = 'running'
                logging.info('Simulation started.')
            else:
                self._environment_failed()
        elif self._state == 'unconnected':
            logging.info("Environment is not connected.")
        #elif self._state == 'running':
            #logging.info("Simulation is already running.")
    def connect(self):
        if self._state == 'unconnected':
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socket_connected = False
            for trial in range(10):
                try:
                    logging.warning(f'Try to connect to the environment docker ({trial+1}/10).')
                    self._socket.connect((self._server_address, self._server_port))
                except socket.error:
                    logging.warning('Connection failed.')
                    time.sleep(0.5)
                else:
                    socket_connected = True
                    break
            if socket_connected:
                self._messenger = Messenger(self._socket)
                self._messenger.send('player_id', self._identifier)
                msg_type, msg = self._messenger.recv()
                if msg_type == 'connection_successful':
                    self._state = 'idle'
                    logging.info('Successfully connected to environment.')
                else:
                    self._environment_failed()
            else:
                logging.warning('Check out if the environment docker is up.')
        else:
            logging.info('Already connected to environment.')
    def run(self, run_time):
        self.start_simulation(time_us=run_time)
        
    def _check_action(self, action: int) -> bool:
        if action == 0:
            return True
        elif action == 1:
            sta_allocation_dict = self.sta_allocation_dict
            if len(sta_allocation_dict) == 0:
                logging.warning("\'sta_allocation_dict\' should not be empty.")
                return False
            for freq_channel in sta_allocation_dict:
                sta = sta_allocation_dict[freq_channel]
                if sta not in self._sta_list:
                    logging.warning(f"Value of \'sta_allocation_dict\' should be in {self._sta_list}.")
                    return False
                if freq_channel not in self._freq_channel_list:
                    logging.warning(f"Key of \'sta_allocation_dict\' should be in {self._freq_channel_list}.")
                    return False
            num_unit_packet = action['num_unit_packet']
            if not isinstance(num_unit_packet, int):
                logging.warning("Value of \'num_unit_packet\' should be an integer.")
                return False
            if action['num_unit_packet'] not in self._num_unit_packet_list:
                logging.warning(f"Value of \'num_unit_packet\' should be one of {self._num_unit_packet_list}.")
                return False
        else:
            logging.warning("Value of \'type\' should be either \'sensing\' or \'tx_data_packet\'.")
            return False
        return True
    def _configure_logging(self, log_type: str, enable: bool):
        if self._state == 'idle':
            self._messenger.send('configure_logging', {'log_type': log_type, 'enable': enable})
            msg_type, msg = self._messenger.recv()
            if msg_type == 'logging_configured':
                logging.info("Logging is successfully configured.")
            else:
                self._environment_failed()
        elif self._state == 'running':
            logging.warning("Logging cannot be configured when the simulator is running.")
        elif self._state == 'unconnected':
            logging.warning("Environment is not connected.")    
    def enable_video_logging(self):
        self._configure_logging(log_type='video', enable=True)
    def disable_video_logging(self):
        self._configure_logging(log_type='video', enable=False)
    def enable_text_logging(self):
        self._configure_logging(log_type='text', enable=True)
    def disable_text_logging(self):
        self._configure_logging(log_type='text', enable=False)
    def _environment_failed(self):
        raise Exception('Environment failed. Restart the environment docker.')    
    def get_score(self) -> Dict:
        return self._score
    def set_init(self):
        initial_action = {'type': 'sensing'}
        observation_dict = self.step(initial_action)
        self._observation = self.convert_observation_dict_to_arr(observation_dict['observation'])
    def convert_observation_dict_to_arr(self, observation):
        observation_type = observation['type']
        observation_arr = np.zeros(self._num_freq_channel)
        if observation_type == 'sensing':
            print(observation)
            is_sensed = observation['sensed_freq_channel']
            observation_arr[is_sensed] = 1
        elif observation_type == 'tx_data_packet':
            observation_arr[:] = 2
            success_freq_channel_list = observation['success_freq_channel']
            observation_arr[success_freq_channel_list] = 3
        return observation_arr

