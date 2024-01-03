import random
import numpy as np

class ReplayMemory:

    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward,constraint,center_pos,next_center_pos, next_state, mask, t=None, next_t=None):

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        self.buffer[self.position] = (state, action, reward,constraint,center_pos,next_center_pos, next_state, mask, t, next_t)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):

        batch = random.sample(self.buffer, batch_size)
        state, action, reward,constraint,center_pos, next_center_pos,next_state, mask, t, next_t = map(np.stack, zip(*batch))
        return state, action, reward,constraint,center_pos,next_center_pos, next_state, mask, t, next_t

    def __len__(self):
        return len(self.buffer)
