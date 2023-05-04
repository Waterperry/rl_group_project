import numpy as np
import pygame
import gym

# this class copied from https://stackoverflow.com/questions/510357/how-to-read-a-single-character-from-the-user
class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch


def play():
    env = gym.envs.make('CarRacing-v2', continuous=False, render_mode='human')
    run_training_loop(1, env)


def run_training_loop(num_episodes=1, env=None):
    returns = np.zeros(num_episodes)
    moves = {' ': 0, 'd': 1, 'a': 2, 'w': 3, 's': 4}
    for ep in range(num_episodes):
        _ = env.reset()
        done, truncated = False, False
        ep_return = 0.
        getter = _GetchUnix()

        while not done and not truncated:
            action = moves[getter()]
            _, reward, done, truncated, _ = env.step(action)

            done = done or truncated

            env.render()
            pygame.event.get()

            ep_return += reward

        returns[ep] = ep_return
        print(f"{ep}, {ep_return}")
