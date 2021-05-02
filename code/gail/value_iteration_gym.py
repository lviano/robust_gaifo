import argparse
import gym
import gym_simple
import os
import sys
import pickle
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *

class PolicyNet:

    def __init__(self):

        self.policy = None

    def select_action(self, state):

        state = np.where(state==1)
        #state = int(state)
        return self.policy[state]



def value_iteration(env, tol=1e-10):
    v = np.zeros(env.n_states)
    q = np.zeros((env.n_states, env.n_actions))
    policy_net = PolicyNet()
    while True:
        v_old = np.copy(v)
        for a in range(env.n_actions):
            q[:, a] = env.R[a, :] + env.gamma * env.sparseP[a].dot(v)
        v = np.max(q, axis=1)
        if np.linalg.norm(v - v_old) < tol:
            break
    policy_net.policy = np.argmax(q, axis=1)
    return policy_net

def main_loop():

    num_steps = 0

    for i_episode in count():

        state = env.reset()
        #s_index = env.state_to_index(state)

        reward_episode = 0

        for t in range(10000):
            action = policy_net.select_action(state)
            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            reward_episode += reward
            num_steps += 1
            expert_traj.append(np.hstack([state, action]))
            state_only_expert_traj.append(np.hstack([state, next_state]))
            if done or num_steps >= args.max_expert_state_num:
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_expert_state_num:
            break

    return expert_traj, state_only_expert_traj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save expert trajectory')
    parser.add_argument('--env-name', default="gridworld-v0", metavar='G',
                        help='It is the only one available')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                        help='maximal number of main iterations (default: 50000)')
    parser.add_argument('--noiseE', type=float, default=None, metavar='G')
    parser.add_argument('--grid-type', type=int, default=None, metavar='N')
    args = parser.parse_args()

    env = gym.make("gridworld-v0", prop=args.noiseE, env_type=args.grid_type)
    subfolder = "env" + str(args.grid_type) + "noiseE" + str(args.noiseE)
    if not os.path.isdir(assets_dir(subfolder + "/expert_traj")):
        os.makedirs(assets_dir(subfolder + "/expert_traj"))
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    is_disc_action = len(env.action_space.shape) == 0
    state_dim = env.observation_space.shape[0]

    expert_traj = []
    state_only_expert_traj = []

    policy_net = value_iteration(env)

    expert_traj, state_only_expert_traj = main_loop()
    expert_traj = np.stack(expert_traj)
    state_only_expert_traj = np.stack(state_only_expert_traj)
    running_state = None
    pickle.dump((expert_traj, running_state), open(os.path.join(assets_dir(subfolder), 'expert_traj/{}_expert_traj_value_iteration.p'.format(args.env_name)), 'wb'))
    pickle.dump((state_only_expert_traj, running_state), open(os.path.join(assets_dir(subfolder), 'expert_traj/{}_state_only_expert_traj_value_iteration.p'.format(args.env_name)), 'wb'))
