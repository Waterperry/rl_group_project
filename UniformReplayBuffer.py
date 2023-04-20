import numpy as np


class Experience:
    def __init__(self, s, done, a, r, s_p):
        self.state = s
        self.done = done
        self.action = a
        self.reward = r
        self.next_state = s_p

        self._as_list = [s, a, r, s_p]
        self._list_iter_counter = 0

    def __iter__(self):
        self._list_iter_counter += 1
        if self._list_iter_counter < 4:
            yield self._as_list[self._list_iter_counter - 1]
        else:
            raise StopIteration

    def is_terminal(self):
        return self.done


class UniformReplayBuffer:
    def __init__(self, max_length=10000, minibatch_size=32):
        self.buf = []
        self.max_len = max_length
        self.mb_size = minibatch_size

    def sample_minibatch(self):
        if len(self.buf) <= self.mb_size:
            raise ValueError("Not enough experiences in the buffer to sample a minibatch."
                             "Consider asserting that `buffer.num_experiences() > buffer.mb_size'")

        mb_start_idx = np.random.randint(0, len(self.buf) - self.mb_size)
        return self.buf[mb_start_idx:mb_start_idx + self.mb_size]

    def add_experience(self, experience: Experience):
        self.buf.append(experience)

    def num_experiences(self):
        return len(self.buf)
