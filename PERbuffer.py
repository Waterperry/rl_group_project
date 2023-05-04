__author__ = 'vv389'

from collections import deque
import numpy as np


class PrioritizedReplayBuffer:

    def __init__(self, buffer_size, alpha=0.6, beta=0.4, batch_size=64, n_steps=1):
        self.buffer_size = buffer_size
        self.alpha = alpha  # prioritize based on the TD error
        self.beta = beta    # weight importance sampling corrections
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        # self.buffer = []
        # self.priorities = []
        self.batch_size = batch_size
        self.n_steps = n_steps

    def add_experience(self, experience):
        self.buffer.append(experience)
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_prio)
      
    def get_batch(self):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha

        # choose a batch of indices with probability probs. If we're doing n-step, don't take from the end of the buffer
        buf_length = len(self.buffer) - (self.n_steps + 1)

        probs = probs[:buf_length]
        probs /= probs.sum()
        indices = np.random.choice(buf_length, self.batch_size, p=probs)

        batch = []
        # indices
        for index in indices:

            experience = self.buffer[index]
            batch.append(experience)

        importance = ((1.0 / len(self.buffer)) * (1.0/probs[indices])) ** self.beta
        importance /= importance.max()
        return batch, (importance, indices)

    def update_priorities(self, indices, errors):
        for i, index in enumerate(indices):
            self.priorities[index] = abs(errors[i]) + 1e-6

    def num_experiences(self):
        return len(self.buffer)
