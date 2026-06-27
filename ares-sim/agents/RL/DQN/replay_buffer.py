import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity: int):
       
        self.memory = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        
        self.memory.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
       
        return random.sample(self.memory, batch_size)

    def __len__(self):
     
        return len(self.memory)
