import argparse
import gym
#import pybulletgym
#import gym_simple
import gym_reach
import os
import sys
import pickle
import time
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_policy import Policy, OpponentPolicy
from models.mlp_critic import Value
from models.mlp_policy_disc import DiscretePolicy
from models.mlp_discriminator import Discriminator
from torch import nn
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='log std for the policy (default: -0.0)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--alpha', type=float, default=0.9, metavar='G',
                    help='mixture parameter alpha*player + (1-alpha)* opponent (default: 0.9)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=4, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--exp-type', type=str, default="mismatch", metavar='N',
                    help="experiment type: noise, friction or mismatch")
parser.add_argument('--alg', type=str, default="gail", metavar='N',
                    help="gail or gaifo")
parser.add_argument('--opponent_steps', type=int, default=1, metavar='N',
                    help="number of opponent PPO epochs")
parser.add_argument('--reward-type', type=str, default="negative", metavar='N',
                    help="functional form of the reward taking as "
                         "input the discriminator output. Options: positive, negative, airl")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--noiseE', type=float, default=None, metavar='G')
parser.add_argument('--noiseL', type=float, default=None, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mulL', type=float, default=1.0, metavar='G',
                    help="Mass Multiplier for learner environment")
parser.add_argument('--len-mulL', type=float, default=1.0, metavar='G',
                    help="Lenght Multiplier for learner environment")
parser.add_argument('--mass-mulE', type=float, default=1.0, metavar='G',
                    help="Mass multiplier for expert environment")
parser.add_argument('--len-mulE', type=float, default=1.0, metavar='G',
                    help="Lenght multiplier for expert environment")
parser.add_argument('--scheduler-lr', action='store_true', default=False,
                    help='Use discriminator lr scheduler')
parser.add_argument('--warm-up', action='store_true', default=False,
                    help='Discriminator Warm UP')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
print(device, "device")
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
max_grad = 40
global subfolder
"""environment"""
reward_type = args.reward_type
if args.alg == "airl":
    reward_type = "airl"
    args.reward_type == "airl"
if args.exp_type == "noise":
    if args.env_name == "gridworld-v0":
        env = gym.make(args.env_name, prop=args.noiseL, env_type=args.grid_type)
        subfolder = "env" + str(args.grid_type) + "noiseE" + str(args.noiseE) + "noiseL" + str(args.noiseL)
    else:
        env = gym.make(args.env_name)
        subfolder = "env" + args.env_name + "noiseE" + str(args.noiseE)

elif args.exp_type == "mismatch":
    env = gym.make(args.env_name)

    if args.env_name == "Acrobot-v1":
        subfolder = "env" + args.env_name + "massL" + str(args.mass_mulL) + "massE" + str(args.mass_mulE) \
                    + "lenL" + str(args.len_mulL) + "lenE" + str(args.len_mulE)
        env.env.LINK_LENGTH_1 *= args.len_mulL
        env.env.LINK_LENGTH_2 *= args.len_mulL
        env.env.LINK_MASS_1 *= args.mass_mulL
        env.env.LINK_MASS_2 *= args.mass_mulL
    elif args.env_name == "CartPole-v1":
        subfolder = "env" + args.env_name + "massL" + str(args.mass_mulL) + "massE" + str(args.mass_mulE) \
                    + "lenL" + str(args.len_mulL) + "lenE" + str(args.len_mulE)
        env.env.masspole *= args.mass_mulL
        env.env.masscart *= args.mass_mulL
        env.env.length *= args.len_mulL
    elif args.env_name == "HalfCheetah-v2" or args.env_name == "Ant-v2" or args.env_name == "Walker2d-v2" or args.env_name == "Hopper-v2" or args.env_name == "Swimmer-v2" or args.env_name == "InvertedDoublePendulum-v2" or args.env_name == "InvertedPendulum-v2":
        env.env.model.body_mass[:] *= args.mass_mulL
        subfolder = "env" + args.env_name + "massL" + str(args.mass_mulL) + "massE" + str(args.mass_mulE)
    elif args.env_name == "gym_reach:reachNoisy-v0":
        env = gym.make("gym_reach:reachNoisy-v0", render_mode='rgb_array',
                       action_noise_mean=0.1,
                       action_noise_var=args.mass_mul)
        subfolder = "env" + args.env_name + "noise_varL" + str(args.mass_mulL) \
                    + "noise_varE" + str(args.mass_mulE)
    elif args.env_name == "MountainCarContinuous-v0":
        env.env.power *= args.mass_mulL
        subfolder = "env" + args.env_name + "powerL" + str(args.mass_mulL) + "powerE1.0"
    elif args.env_name == "LunarLanderContinuous-v2":
        gym.envs.box2d.lunar_lander.MAIN_ENGINE_POWER = gym.envs.box2d.lunar_lander.MAIN_ENGINE_POWER * args.mass_mulL
        env = gym.make(args.env_name)
        subfolder = "env" + args.env_name + "powerL" + str(args.mass_mulL) + "powerE1.0"
elif args.exp_type == "friction":
    env = gym.make(args.env_name)
    env.env.model.geom_friction[:] *= args.mass_mulL
    subfolder = "env" + args.env_name + "frictionL" + str(args.mass_mulL) + "frictionE" + str(args.mass_mulE)

if not os.path.isdir(assets_dir(subfolder + "/learned_models")):
    os.makedirs(assets_dir(subfolder + "/learned_models"))
if not os.path.isdir(assets_dir(subfolder + "/reward_history")):
    os.makedirs(assets_dir(subfolder + "/reward_history"))

state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0

action_dim = 1 if is_disc_action else env.action_space.shape[0]
running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if is_disc_action:
    policy_net = DiscretePolicy(state_dim, env.action_space.n)
    opponent_net = DiscretePolicy(state_dim, env.action_space.n)
else:
    policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    opponent_net = OpponentPolicy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    if args.env_name == "Ant-v2":
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std, hidden_size=(256, 256))
        opponent_net = OpponentPolicy(state_dim, env.action_space.shape[0], log_std=args.log_std,
                                      hidden_size=(256, 256))

"""define discriminator"""
if args.alg == "gaifo" or args.alg == "airl":
    if args.env_name == "HalfCheetah-v2" or args.env_name == "Ant-v2":
        discrim_net = Discriminator(state_dim + state_dim, hidden_size=(400, 300))

    else:
        discrim_net = Discriminator(state_dim + state_dim)
elif args.alg == "gail":
    discrim_net = Discriminator(state_dim + env.action_space.n) \
        if is_disc_action else Discriminator(state_dim + env.action_space.shape[0])

value_net = Value(state_dim)
discrim_criterion = nn.BCEWithLogitsLoss()
to_device(device, policy_net, opponent_net, value_net, discrim_net, discrim_criterion)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_opponent = torch.optim.Adam(opponent_net.parameters(), lr=args.learning_rate)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=args.learning_rate)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)

if args.scheduler_lr:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_discrim,
                                                step_size=1000, gamma=0.5)

# optimization epoch number and batch size for PPO
optim_epochs = 20  # 10
optim_batch_size = 64
state_only = True if (args.alg == "gaifo" or args.alg == "airl") else False

# load trajectory
expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
if not running_state is None:
    running_state.fix = True
else:
    running_state = lambda x: x
if is_disc_action and args.alg == "gail":
    one_hot_actions = to_categorical(expert_traj[:, -1].astype(int), env.action_space.n)

    expert_traj = np.concatenate([expert_traj[:, :-1], one_hot_actions], axis=1)


def expert_reward(state, next, reward_type):
    if args.alg == "gaifo" or args.alg == "airl":
        input_discrim = tensor(np.hstack([state, next]), dtype=dtype)

    elif args.alg == "gail":
        if isinstance(next, int):
            next = torch.from_numpy(to_categorical(next, env.action_space.n)).to(dtype).to(device)
        input_discrim = tensor(np.hstack([state, next]), dtype=dtype)
    with torch.no_grad():
        if reward_type == "airl":
            return math.log(1 - torch.sigmoid(discrim_net(input_discrim))[0].item() + 1e-8) \
                   - math.log(torch.sigmoid(discrim_net(input_discrim))[0].item() + 1e-8)
        if reward_type == "negative":
            return math.log(1 - torch.sigmoid(discrim_net(input_discrim))[0].item() + 1e-8)
        if reward_type == "positive":
            return -math.log(torch.sigmoid(discrim_net(input_discrim))[0].item() + 1e-8)


"""create agent"""
agent = Agent(env, policy_net, device, custom_reward=expert_reward,
              running_state=running_state, render=args.render, num_threads=args.num_threads,
              state_only=state_only, opponent_net=opponent_net, alpha=args.alpha,
              reward_type=reward_type)


def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    next_states = torch.from_numpy(np.stack(batch.next_state)).to(dtype).to(device)
    player_actions = torch.from_numpy(np.stack(batch.player_action)).to(dtype).to(device)
    opponent_actions = torch.from_numpy(np.stack(batch.opponent_action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, player_actions)
        opponent_fixed_log_probs = opponent_net.get_log_prob(states, opponent_actions)
    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """update discriminator"""
    expert_discrim_input = torch.from_numpy(expert_traj).to(dtype).to(device)

    if i_iter < 10 and args.warm_up:
        discrim_epoch = 10
    else:
        discrim_epoch = 1
    for _ in range(discrim_epoch):
        if args.alg == "gail":
            if is_disc_action:
                actions_g = torch.from_numpy(to_categorical(actions.detach().numpy().astype(int),
                                                            env.action_space.n)).to(dtype).to(device)
                g_o = discrim_net(torch.cat([states, actions_g], 1))
            else:
                g_o = discrim_net(torch.cat([states, actions], 1))
        elif args.alg == "gaifo" or args.alg == "airl":
            g_o = discrim_net(torch.cat([states, next_states], 1))
        e_o = discrim_net(expert_discrim_input)
        optimizer_discrim.zero_grad()
        discrim_loss = discrim_criterion(g_o, ones((states.shape[0], 1), device=device)) + \
                       discrim_criterion(e_o, zeros((expert_traj.shape[0], 1), device=device))
        discrim_loss.backward()
        if args.scheduler_lr:
            scheduler.step()
        optimizer_discrim.step()

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, player_actions, opponent_actions, returns, advantages, fixed_log_probs, opponent_fixed_log_probs = \
            states[perm].clone(), player_actions[perm].clone(), opponent_actions[perm].clone(), returns[perm].clone(), \
            advantages[perm].clone(), \
            fixed_log_probs[perm].clone(), opponent_fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, player_actions_b, opponent_actions_b, advantages_b, returns_b, fixed_log_probs_b, opponent_fixed_log_probs_b = \
                states[ind], player_actions[ind], opponent_actions[ind], advantages[ind], returns[ind], fixed_log_probs[
                    ind], \
                opponent_fixed_log_probs[ind]

            # Update the player
            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, player_actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg, max_grad=max_grad)
            # Update the opponent
            ppo_step(opponent_net, value_net, optimizer_opponent, optimizer_value, 1, states_b, opponent_actions_b,
                     returns_b,
                     advantages_b, opponent_fixed_log_probs_b, args.clip_epsilon, args.l2_reg, opponent=True,
                     max_grad=max_grad)

    """#Update the opponent
    for _ in range(args.opponent_steps):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, player_actions, opponent_actions, returns, advantages, fixed_log_probs, opponent_fixed_log_probs = \
            states[perm].clone(), player_actions[perm].clone(), opponent_actions[perm].clone(), returns[perm].clone(), \
            advantages[perm].clone(), \
            fixed_log_probs[perm].clone(), opponent_fixed_log_probs[perm].clone()
        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, player_actions_b, opponent_actions_b, advantages_b, returns_b, fixed_log_probs_b, opponent_fixed_log_probs_b = \
                states[ind], player_actions[ind], opponent_actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind],\
                opponent_fixed_log_probs[ind]
            ppo_step(opponent_net, value_net, optimizer_opponent, optimizer_value, 1, states_b, opponent_actions_b, returns_b,
                 advantages_b, opponent_fixed_log_probs_b, args.clip_epsilon, args.l2_reg, opponent = True, max_grad=max_grad)
    """


def main_loop():
    rewards = []
    best_reward = -10000
    for i_iter in range(args.max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        discrim_net.to(torch.device('cpu'))
        batch, log = agent.collect_samples(args.min_batch_size)
        discrim_net.to(device)

        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\texpert_R_avg {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1 - t0, log['avg_c_reward'], log['avg_reward']))
            rewards.append(log['avg_reward'])
            pickle.dump(rewards, open(
                os.path.join(assets_dir(subfolder), 'reward_history/{}_{}_{}.p'.format(args.env_name
                                                                                       + str(args.seed), args.alg,
                                                                                       args.alpha)), 'wb'))

        if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net, discrim_net)
            pickle.dump((policy_net, value_net, discrim_net), open(os.path.join(assets_dir(subfolder),
                                                                                'learned_models/{}_{}_{}.p'.format(
                                                                                    args.env_name + str(args.seed),
                                                                                    args.alg, args.alpha)), 'wb'))
            if log['avg_reward'] > best_reward:
                print(best_reward)
                pickle.dump((policy_net, value_net, discrim_net),
                            open(os.path.join(assets_dir(subfolder),
                                              'learned_models/{}_{}_best_{}.p'.format(
                                                  args.env_name + str(args.seed), args.alg, args.alpha)), 'wb'))
                best_reward = copy.deepcopy(log['avg_reward'])

            to_device(device, policy_net, value_net, discrim_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()
