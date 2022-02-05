import numpy as np

class Buffer():

    def __init__(self, max_buffer_size=200):
        self.buffer_counter = 0
        self.max_buffer_size = max_buffer_size
        self.observation_fine = np.array([])
        self.observation_coarse = np.array([])
        self.new_observation_fine = np.array([])
        self.new_observation_coarse = np.array([])
        self.encoding_fine = np.array([])
        self.encoding_coarse = np.array([])
        self.new_encoding_fine = np.array([])
        self.new_encoding_coarse = np.array([])
        self.reward = np.array([])
        self.action = np.array([])

    def clear(self):
        self.buffer_counter = 0
        self.observation_fine = np.array([])
        self.observation_coarse = np.array([])
        self.new_observation_fine = np.array([])
        self.new_observation_coarse = np.array([])
        self.encoding_fine = np.array([])
        self.encoding_coarse = np.array([])
        self.new_encoding_fine = np.array([])
        self.new_encoding_coarse = np.array([])
        self.reward = np.array([])
        self.action = np.array([])

    def store(self, observation_fine, observation_coarse, new_observation_fine, new_observation_coarse, 
              encoding_fine, encoding_coarse, new_encoding_fine, new_encoding_coarse, action,reward):
        if self.buffer_counter == 0:
            self.observation_fine = np.array([observation_fine])
            self.observation_coarse = np.array([observation_coarse])
            self.new_observation_fine = np.array([new_observation_fine])
            self.new_observation_coarse = np.array([new_observation_fine])
            self.encoding_fine = np.array([encoding_fine.detach().numpy()])
            self.encoding_coarse = np.array([encoding_coarse.detach().numpy()])
            self.new_encoding_fine = np.array([new_encoding_fine.detach().numpy()])
            self.new_encoding_coarse = np.array([new_encoding_coarse.detach().numpy()])
            self.reward = np.array([reward])
            self.action = np.array([action])
        else: 
            self.observation_fine = np.concatenate((self.observation_fine, [observation_fine]), axis=0)
            self.observation_coarse = np.concatenate((self.observation_coarse, [observation_coarse]), axis=0)
            self.new_observation_fine = np.concatenate((self.new_observation_fine, [new_observation_fine]), axis=0)
            self.new_observation_coarse = np.concatenate((self.new_observation_coarse, [new_observation_coarse]), axis=0)
            self.encoding_fine = np.concatenate((self.encoding_fine, [encoding_fine.detach().numpy()]), axis=0)
            self.encoding_coarse = np.concatenate((self.encoding_coarse, [encoding_coarse.detach().numpy()]), axis=0)
            self.new_encoding_fine = np.concatenate((self.new_encoding_fine, [new_encoding_fine.detach().numpy()]), axis=0)
            self.new_encoding_coarse = np.concatenate((self.new_encoding_coarse, [new_encoding_coarse.detach().numpy()]), axis=0)
            self.reward = np.concatenate((self.reward, [reward]), axis=0)
            self.action = np.concatenate((self.action, [action]), axis=0)
        self.buffer_counter += 1

    def concat(self, process_buffer):
        if self.buffer_counter == 0:
            self.observation_fine = np.array(process_buffer.observation_fine)
            self.observation_coarse = np.array(process_buffer.observation_coarse)
            self.new_observation_fine = np.array(process_buffer.new_observation_fine)
            self.new_observation_coarse = np.array(process_buffer.new_observation_fine)
            self.encoding_fine = np.array(process_buffer.encoding_fine)
            self.encoding_coarse = np.array(process_buffer.encoding_coarse)
            self.new_encoding_fine = np.array(process_buffer.new_encoding_fine)
            self.new_encoding_coarse = np.array(process_buffer.new_encoding_coarse)
            self.reward = np.array(process_buffer.reward)
            self.action = np.array(process_buffer.action)
        else:
            # concatenate process buffer to end of global buffer:
            self.observation_fine = np.concatenate((self.observation_fine, process_buffer.observation_fine), axis=0)
            self.observation_coarse = np.concatenate((self.observation_coarse, process_buffer.observation_coarse), axis=0)
            self.new_observation_fine = np.concatenate((self.new_observation_fine, process_buffer.new_observation_fine), axis=0)
            self.new_observation_coarse = np.concatenate((self.new_observation_coarse, process_buffer.new_observation_coarse), axis=0)
            self.encoding_fine = np.concatenate((self.encoding_fine, process_buffer.encoding_fine), axis=0)
            self.encoding_coarse = np.concatenate((self.encoding_coarse, process_buffer.encoding_coarse), axis=0)
            self.new_encoding_fine = np.concatenate((self.new_encoding_fine, process_buffer.new_encoding_fine), axis=0)
            self.new_encoding_coarse = np.concatenate((self.new_encoding_coarse, process_buffer.new_encoding_coarse), axis=0)
            self.reward = np.concatenate((self.reward, process_buffer.reward), axis=0)
            self.action = np.concatenate((self.action, process_buffer.action), axis=0)
            # clip global buffer to max_buffer_size:
            self.observation_fine = self.observation_fine[-self.max_buffer_size:]
            self.observation_coarse = self.observation_coarse[-self.max_buffer_size:]
            self.new_observation_fine = self.new_observation_fine[-self.max_buffer_size:]
            self.new_observation_coarse = self.new_observation_coarse[-self.max_buffer_size:]
            self.encoding_fine = self.encoding_fine[-self.max_buffer_size:]
            self.encoding_coarse = self.encoding_coarse[-self.max_buffer_size:]
            self.new_encoding_fine = self.new_encoding_fine[-self.max_buffer_size:]
            self.new_encoding_coarse = self.new_encoding_coarse[-self.max_buffer_size:]
            self.reward = self.reward[-self.max_buffer_size:]
            self.action = self.action[-self.max_buffer_size:]
        self.buffer_counter += 1