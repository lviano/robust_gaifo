import pickle
import sys
import argparse
import copy
import ast
import os
import gym
import gym_simple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import count
from utils import *
from utils import plot
from gail.value_iteration_gym import PolicyNet, value_iteration
parser = argparse.ArgumentParser()
parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument("--alg", nargs='+', default=["0"])
parser.add_argument("--alpha", nargs='+', default=["0"])
parser.add_argument("--seeds", nargs='+', default=["0"])
parser.add_argument("--noiseE", nargs='+', default=["0"],
                    help='expert noise')
parser.add_argument("--mass-muls", nargs='+', default=["0"],
                    help='expert masses')
parser.add_argument("--len-muls", nargs='+', default=["0"],
                    help='expert lengths')
parser.add_argument("--noiseL", type = float, metavar='G',
                    help='learner noise')
parser.add_argument("--grid-type", type = int, metavar='G',
                    help='learner noise')
parser.add_argument("--render", default= False, action='store_true')
parser.add_argument("--no-compute", default= False, action='store_true')
parser.add_argument("--var-mass", default= False, action='store_true')
parser.add_argument("--var-len", default= False, action='store_true')
parser.add_argument("--vi-expert", default= False, action='store_true')
parser.add_argument("--best", default= False, action='store_true')
parser.add_argument("--max-steps", type=int, default=5000)
args = parser.parse_args()

def evaluate_loop(policy_net, running_state, expert_flag):

    num_steps = 0
    episodes_reward = []

    for i_episode in count():

        state = env.reset()
        state_expert = copy.deepcopy(state)
        """if args.env_name == "gridworld-v0":
            s_index_expert = env.state_to_index(state_expert)"""
        state = running_state(state)
        reward_episode = 0

        for t in range(10000):
            state_var = tensor(state).unsqueeze(0).to(dtype)
            # choose mean action
            if expert_flag and args.env_name == "gridworld-v0":
                """if args.env_name == "gridworld-v0":
                    action = policy_net.select_action(s_index_expert)
                else:"""
                action = policy_net.select_action(state_expert)
            else:
                if not is_disc_action:
                    action = policy_net(state_var)[0][0].detach().numpy()
                else:
                    action = policy_net.select_action(state_var)[0].numpy()
            # action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            next_state, reward, done, _ = env.step(action)
            state_expert = copy.deepcopy(next_state)
            """if args.env_name == "gridworld-v0":
                s_index_expert = env.state_to_index(state_expert)"""
            next_state = running_state(next_state)
            reward_episode += reward
            num_steps += 1


            if args.render:
                env.render()
            if done or num_steps >= args.max_steps:

                break

            state = next_state

        episodes_reward.append(reward_episode)
        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))

        if num_steps >= args.max_steps:
            break

    return np.mean(episodes_reward)


dtype = torch.float64
torch.set_default_dtype(dtype)
if not args.no_compute:
    to_plot = []
    to_plot_std = []
    if "expert" in args.alg and not args.alg[0] == "expert":
        raise ValueError("expert must be passed as first argument in --alg")
    if "gail" in args.alg and not args.alg[1] == "gail":
        raise ValueError("gail must be passed as second argument in --alg")

    if args.env_name == "gridworld-v0":
        env = gym.make("gridworld-v0", prop=args.noiseL, env_type=args.grid_type)
        subfolder = "env"+str(args.grid_type)+"noiseL"+str(args.noiseL)
        if not os.path.isdir(assets_dir(subfolder)):
            os.makedirs(assets_dir(subfolder))

        env.seed(0)
        torch.manual_seed(0)

        is_disc_action = len(env.action_space.shape) == 0
        state_dim = env.observation_space.shape[0]

        if not os.path.isdir("../plot/" + subfolder):
            os.makedirs("../plot/" + subfolder)

        gail_done = False
        expert_done = False
        for alg in args.alg:

            for alpha in args.alpha:
                to_append = []
                to_append_std = []
                for noiseE in args.noiseE:
                    means_avg = []
                    stds_avg = []
                    if args.vi_expert:
                        _, running_state = pickle.load(
                            open(os.path.join(assets_dir("env" + str(args.grid_type) + "noiseE" + str(noiseE)),
                                              'expert_traj/' + args.env_name + "_expert_traj_value_iteration.p"), "rb"))
                    else:
                        _, running_state = pickle.load(
                            open(os.path.join(assets_dir("env" + str(args.grid_type) + "noiseE" + str(noiseE)),
                                              'expert_traj/' + args.env_name + "_expert_traj.p"), "rb"))
                    for seed in args.seeds:
                        means = []
                        data_subfolder = "env" + str(args.grid_type) + "noiseE" + str(noiseE) + "noiseL" + str(
                            args.noiseL)
                        if alg == "robust_gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                                 'learned_models/{}_robust_gail_{}.p'.format(
                                                                                     args.env_name + str(seed),
                                                                                     alpha))) > 0 and not args.best:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_robust_gail_{}.p'.format(
                                                                                 args.env_name + str(seed), alpha)),
                                                                "rb"))
                            expert_flag = False
                        elif alg == "robust_gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                                 'learned_models/{}_robust_gail_best_{}.p'.format(
                                                                                     args.env_name + str(seed),
                                                                                     alpha))) > 0 and args.best:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_robust_gail_best_{}.p'.format(
                                                                                 args.env_name + str(seed), alpha)),
                                                                "rb"))
                            expert_flag = False
                        elif alg == "gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                            'learned_models/{}_gail.p'.format(
                                                                                args.env_name + str(seed)))) > 0 and not args.best:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_gail.p'.format(
                                                                                 args.env_name + str(seed))), "rb"))
                            expert_flag = False
                        elif alg == "gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                            'learned_models/{}_gail_best.p'.format(
                                                                                args.env_name + str(seed)))) > 0 and args.best:

                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_gail_best.p'.format(
                                                                                 args.env_name + str(seed))), "rb"))
                            expert_flag = False
                        elif alg == "expert":
                            env_e = gym.make("gridworld-v0", prop=float(noiseE), env_type=args.grid_type)
                            policy_net = value_iteration(env_e)
                            expert_flag = True
                        if not running_state is None:
                            running_state.fix = True
                        else:
                            running_state = lambda x: x
                        for _ in range(10):
                            mean_reward = evaluate_loop(policy_net, running_state, expert_flag)
                            means.append(mean_reward)

                        means_avg.append(np.mean(means))
                        #stds_avg.append(np.std(means))

                    to_append.append(np.mean(means_avg))
                    #to_append_std.append(np.mean(stds_avg))
                    to_append_std.append(np.std(means_avg))
                if (alg == "gail" and gail_done) or (alg == "expert" and expert_done):
                    break
                to_append = np.array(to_append)
                to_append_std = np.array(to_append_std)
                to_plot.append(to_append)
                to_plot_std.append(to_append_std)
                if alg == "gail":
                    gail_done = True
                if alg == "expert":
                    expert_done = True
        base_names = [alg_name for alg_name in ["expert", "gail"] if alg_name in args.alg]
        names = base_names + args.alpha if base_names else args.alpha
        if not args.best:
            pickle.dump((to_plot, to_plot_std, names), open("../plot/" + subfolder + "/DataCompareAlphas"
                                                        + args.env_name + str(args.grid_type) + "noiseL" + str(
            args.noiseL) +str(args.seeds)+".p", 'wb'))
        else:
            pickle.dump((to_plot, to_plot_std, names), open("../plot/" + subfolder + "/DataCompareAlphas"
                                                            + args.env_name + str(args.grid_type) + "noiseL" + str(
                args.noiseL) + str(args.seeds) +"best.p", 'wb'))

    elif args.env_name == "CartPole-v1" or args.env_name == "Acrobot-v1":
        env = gym.make(args.env_name)
        if args.var_mass:
            subfolder = "env" + args.env_name + "massL1.0"
        elif args.var_len:
            subfolder = "env" + args.env_name + "lenL1.0"
        if not os.path.isdir(assets_dir(subfolder)):
            os.makedirs(assets_dir(subfolder))

        env.seed(0)
        torch.manual_seed(0)

        is_disc_action = len(env.action_space.shape) == 0
        state_dim = env.observation_space.shape[0]

        if not os.path.isdir("../plot/" + subfolder):
            os.makedirs("../plot/" + subfolder)

        gail_done = False
        expert_done = False
        if args.var_mass:
            env_features = args.mass_muls
        elif args.var_len:
            env_features = args.len_muls

        for alg in args.alg:

            for alpha in args.alpha:
                to_append = []
                to_append_std = []
                for env_f in env_features:
                    means = []
                    if args.var_mass:
                        _, running_state = pickle.load(
                            open(os.path.join(assets_dir("env" + str(args.env_name) + "mass" + str(env_f) + "len1.0"),
                                              'expert_traj/' + args.env_name + "_expert_traj.p"), "rb"))

                    elif args.var_len:
                        _, running_state = pickle.load(
                            open(os.path.join(assets_dir("env" + str(args.env_name) + "mass1.0len" + str(env_f)),
                                              'expert_traj/' + args.env_name + "_expert_traj.p"), "rb"))
                    for seed in args.seeds:
                        if args.var_mass:
                            data_subfolder = "env" + str(args.env_name) + "massL1.0massE" + str(env_f) + "lenL1.0lenE1.0"
                        elif args.var_len:
                            data_subfolder = "env" + str(args.env_name) + "massL1.0massE1.0lenL1.0lenE" + str(env_f)

                        if alg == "robust_gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                                 'learned_models/{}_robust_gail_{}.p'.format(
                                                                                     args.env_name + str(seed),
                                                                                     alpha))) > 0:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_robust_gail_{}.p'.format(
                                                                                 args.env_name + str(seed), alpha)),
                                                                "rb"))
                            expert_flag = False
                        elif alg == "gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                            'learned_models/{}_gail.p'.format(
                                                                                args.env_name + str(seed)))) > 0:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_gail.p'.format(
                                                                                 args.env_name + str(seed))), "rb"))
                            expert_flag = False
                        elif alg == "expert":
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir("env" + str(args.env_name) + "mass1.0len1.0"),
                                              'learned_models/' + args.env_name + "_ppo.p"), "rb"))
                            expert_flag = True
                        if not running_state is None:
                            running_state.fix = True
                        else:
                            running_state = lambda x: x
                        for _ in range(10):
                            print(alg)
                            mean_reward = evaluate_loop(policy_net, running_state, expert_flag)
                            means.append(mean_reward)
                    stds = np.std(means)
                    means = np.mean(means)

                    to_append.append(means)
                    to_append_std.append(stds)
                if alg == "gail" and gail_done or (alg == "expert" and expert_done):
                    break
                if not alg == "expert":
                    to_append = np.array(to_append)
                    to_append_std = np.array(to_append_std)
                    to_plot.append(to_append)
                    to_plot_std.append(to_append_std)
                else:
                    to_append = np.array(to_append)
                    ones = np.ones_like(to_append)
                    to_append = np.mean(to_append)*ones
                    to_append_std = np.mean(to_append_std)*ones
                    to_plot.append(to_append)
                    to_plot_std.append(to_append_std)
                if alg == "gail":
                    gail_done = True
                if alg == "expert":
                    expert_done = True
        base_names = [alg_name for alg_name in ["expert", "gail"] if alg_name in args.alg]
        names = base_names + args.alpha if base_names else args.alpha
        #names = ["gail"] + args.alpha if "gail" in args.alg else args.alpha
        if args.var_len:
            pickle.dump((to_plot, to_plot_std, names), open("../plot/" + subfolder + "/DataCompareAlphas"
                                                            + args.env_name + "lenL1.0.p", 'wb'))
        elif args.var_mass:
            pickle.dump((to_plot, to_plot_std, names), open("../plot/" + subfolder + "/DataCompareAlphas"
                                                        + args.env_name + "massL1.0.p", 'wb'))
    elif args.env_name == "Ant-v2" or args.env_name == "HalfCheetah-v2":
        env = gym.make(args.env_name)
        subfolder = "env"+str(args.env_name)
        if not os.path.isdir(assets_dir(subfolder)):
            os.makedirs(assets_dir(subfolder))

        env.seed(0)
        torch.manual_seed(0)

        is_disc_action = len(env.action_space.shape) == 0
        state_dim = env.observation_space.shape[0]

        if not os.path.isdir("../plot/" + subfolder):
            os.makedirs("../plot/" + subfolder)

        gail_done = False
        expert_done = False
        for alg in args.alg:

            for alpha in args.alpha:
                to_append = []
                to_append_std = []
                for noiseE in args.noiseE:
                    means_avg = []
                    stds_avg = []
                    _, running_state = pickle.load(
                    open(os.path.join(assets_dir("env" + args.env_name + "unif_noise" + str(noiseE)),
                                              'expert_traj/' + args.env_name + "_expert_traj.p"), "rb"))
                    for seed in args.seeds:
                        means = []
                        data_subfolder = "env" +args.env_name + "noiseE" + str(noiseE)

                        if alg == "robust_gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                                 'learned_models/{}_robust_gail_{}.p'.format(
                                                                                     args.env_name + str(seed),
                                                                                     alpha))) > 0 and not args.best:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_robust_gail_{}.p'.format(
                                                                                 args.env_name + str(seed), alpha)),
                                                                "rb"))
                            expert_flag = False
                        elif alg == "robust_gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                                 'learned_models/{}_robust_gail_best_{}.p'.format(
                                                                                     args.env_name + str(seed),
                                                                                     alpha))) > 0 and args.best:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_robust_gail_best_{}.p'.format(
                                                                                 args.env_name + str(seed), alpha)),
                                                                "rb"))
                            expert_flag = False
                        elif alg == "gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                            'learned_models/{}_gail.p'.format(
                                                                                args.env_name + str(seed)))) > 0 and not args.best:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_gail.p'.format(
                                                                                 args.env_name + str(seed))), "rb"))
                            expert_flag = False
                        elif alg == "gail" and os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                                            'learned_models/{}_gail_best.p'.format(
                                                                                args.env_name + str(seed)))) > 0 and args.best:

                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_gail_best.p'.format(
                                                                                 args.env_name + str(seed))), "rb"))
                            expert_flag = False
                        elif alg == "expert":
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir("learned_models"), args.env_name + "_ppo.p"), "rb"))
                            expert_flag = True
                        if not running_state is None:
                            running_state.fix = True
                        else:
                            running_state = lambda x: x
                        for _ in range(10):
                            mean_reward = evaluate_loop(policy_net, running_state, expert_flag)
                            means.append(mean_reward)

                        means_avg.append(np.mean(means))
                        stds_avg.append(np.std(means))

                    to_append.append(np.mean(means_avg))
                    to_append_std.append(np.mean(stds_avg))
                    #to_append_std.append(np.std(means_avg))
                if (alg == "gail" and gail_done) or (alg == "expert" and expert_done):
                    break
                if not alg == "expert":
                    to_append = np.array(to_append)
                    to_append_std = np.array(to_append_std)
                    to_plot.append(to_append)
                    to_plot_std.append(to_append_std)
                else:
                    to_append = np.array(to_append)
                    ones = np.ones_like(to_append)
                    to_append = np.mean(to_append)*ones
                    to_append_std = np.mean(to_append_std)*ones
                    to_plot.append(to_append)
                    to_plot_std.append(to_append_std)
                if alg == "gail":
                    gail_done = True
                if alg == "expert":
                    expert_done = True
        base_names = [alg_name for alg_name in ["expert", "gail"] if alg_name in args.alg]
        names = base_names + args.alpha if base_names else args.alpha
        if not args.best:
            pickle.dump((to_plot, to_plot_std, names), open("../plot/" + subfolder + "/DataCompareAlphas"
                                                        + args.env_name +str(args.seeds)+".p", 'wb'))
        else:
            pickle.dump((to_plot, to_plot_std, names), open("../plot/" + subfolder + "/DataCompareAlphas"
                                                            + args.env_name + str(args.seeds) +"best.p", 'wb'))

    else:
        env = gym.make(args.env_name)
        subfolder = None

        raise ValueError("TODO: Implement for environments other than gridworld, CartPole and Acrobot")

else:
    if args.env_name == "gridworld-v0":
        subfolder = "env"+str(args.grid_type)+"noiseL"+str(args.noiseL)
    elif args.env_name == "HalfCheetah-v2" or args.env_name == "Ant-v2":
        subfolder = "env" + str(args.env_name)
    else:
        subfolder = None
    to_plot_load, to_plot_std_load, names_load = pickle.load(open("../plot/"+subfolder+"/DataCompareAlphas"
                + args.env_name + str(args.grid_type) + "noiseL" + str(args.noiseL)+".p", 'rb'))
    names = ["gail"] + args.alpha if "gail" in args.alg else args.alpha

    to_plot = [p for i, p in enumerate(to_plot_load) if names_load[i] in names ]
    to_plot_std = [p for i, p in enumerate(to_plot_std_load) if names_load[i] in names]
if args.env_name == "gridworld-v0":
    plot.plot_lines_and_ranges(list_to_plot=to_plot,
                          list_sigmas=to_plot_std,
                          list_name= names,
                          axis_label=["Noise E", "Total Reward"],
                          folder="../plot/"+subfolder+"/",
                          title="CompareAlphas" + args.env_name + str(args.grid_type) + "noiseL" + str(args.noiseL) + "best" + str(args.best)+str(args.seeds),
                          x_axis=args.noiseE)

if args.env_name == "Ant-v2" or args.env_name == "HalfCheetah-v2":
    plot.plot_lines_and_ranges(list_to_plot=to_plot,
                          list_sigmas=to_plot_std,
                          list_name= names,
                          axis_label=["Noise E", "Total Reward"],
                          folder="../plot/"+subfolder+"/",
                          title="CompareAlphas" + args.env_name + "best" + str(args.best)+str(args.seeds),
                          x_axis=args.noiseE)

elif args.env_name == "CartPole-v1" or args.env_name == "Acrobot-v1":
    if args.var_len:
        axis_label = ["Length","Total Reward"]
        title = "CompareAlphas" + args.env_name + "lenL1.0"
        x_axis = args.len_muls
    if args.var_mass:
        axis_label = ["Mass","Total Reward"]
        title = "CompareAlphas" + args.env_name + "massL1.0"
        x_axis = args.mass_muls
    plot.plot_lines_and_ranges(list_to_plot=to_plot,
                               list_sigmas=to_plot_std,
                               list_name=names,
                               axis_label=axis_label,
                               folder="../plot/" + subfolder + "/",
                               title=title,
                               x_axis=x_axis)

