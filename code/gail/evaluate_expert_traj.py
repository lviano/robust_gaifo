import argparse
import gym
import pybulletgym
import gym_simple
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from itertools import count
from utils import *


parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--traj-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
parser.add_argument('--noiseE', type=float, default=None, metavar='G')
parser.add_argument('--unif-noise', type=float, default=0, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot masses")
parser.add_argument('--len-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot lengths")
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
if args.env_name == "gridworld-v0":
    env = gym.make("gridworld-v0", prop = args.noiseE, env_type = args.grid_type)
    subfolder = "env"+str(args.grid_type)+"noiseE"+str(args.noiseE)
    if not os.path.isdir(assets_dir(subfolder+"/expert_traj")):
        os.makedirs(assets_dir(subfolder+"/expert_traj"))
elif args.env_name == "CartPole-v1" or args.env_name == "Acrobot-v1":
    env = gym.make(args.env_name)
    subfolder = "env"+ args.env_name + "mass" + str(args.mass_mul)+ "len" + str(args.len_mul)
    if not os.path.isdir(assets_dir(subfolder+"/expert_traj")):
        os.makedirs(assets_dir(subfolder+"/expert_traj"))
    if args.env_name == "Acrobot-v1":
        env.env.LINK_LENGTH_1 *= args.len_mul
        env.env.LINK_LENGTH_2 *= args.len_mul
        env.env.LINK_MASS_1 *= args.mass_mul
        env.env.LINK_MASS_2 *= args.mass_mul
    elif args.env_name == "CartPole-v1":
        env.env.masspole *= args.mass_mul
        env.env.masscart *= args.mass_mul
        env.env.length *= args.len_mul
elif args.env_name == "HalfCheetah-v2" or args.env_name == "Walker2d-v2":
    env = gym.make(args.env_name)
    env.env.model.body_mass[:] *= args.mass_mul
    subfolder = "env"+ args.env_name + "mass" + str(args.mass_mul)
    if not os.path.isdir(assets_dir(subfolder+"/expert_traj")):
        os.makedirs(assets_dir(subfolder+"/expert_traj"))
else:
    env = gym.make(args.env_name)
    subfolder = "env" + args.env_name +"unif_noise"+str(args.unif_noise)
    if not os.path.isdir(assets_dir(subfolder+"/expert_traj")):
        os.makedirs(assets_dir(subfolder+"/expert_traj"))
env.seed(args.seed)
torch.manual_seed(args.seed)
env.reset()
is_disc_action = len(env.action_space.shape) == 0
state_dim = env.observation_space.shape[0]
expert_traj, running_state = pickle.load(open(os.path.join(assets_dir(subfolder), 'expert_traj/{}_expert_traj.p'.format(args.env_name)), 'rb'))
episode_reward=0
episode_rewards = []
for state_action in expert_traj:
    state = state_action[:state_dim]
    action = state_action[state_dim:]
    _, reward, done, _ = env.step(action)
    episode_reward += reward
    if done:
        print(episode_reward)
        episode_rewards.append(episode_reward)
        episode_reward = 0
        env.reset()
print("Mean " + str(np.mean(episode_rewards)) + "+-" + str(np.std(episode_rewards)))
print(expert_traj.shape[0])
env.reset()
random_reward = 0
done=False
while not done:
    _, reward, done, _ = env.step(np.random.uniform(low=-1.0, high=-1.0, size=env.action_space.shape))
    random_reward += reward

print(random_reward)
