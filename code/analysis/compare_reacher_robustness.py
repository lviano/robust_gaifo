import pickle
import sys
import argparse
import copy
import ast
import os
import gym
import gym_reach

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from itertools import count
from utils import *
from utils import plot
from gail.value_iteration_gym import PolicyNet, value_iteration
from evaluate_loop import evaluate_loop

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
parser.add_argument("--noiseL", type=float, metavar='G',
                    help='learner noise')
parser.add_argument("--grid-type", type=int, metavar='G',
                    help='learner noise')
parser.add_argument("--render", default=False, action='store_true')
parser.add_argument("--no-compute", default=False, action='store_true')
parser.add_argument("--var-mass", default=False, action='store_true')
parser.add_argument("--var-len", default=False, action='store_true')
parser.add_argument("--vi-expert", default=False, action='store_true')
parser.add_argument("--best", default=False, action='store_true')
parser.add_argument('--exp-type', type=str, default="mismatch", metavar='N',
                    help="experiment type: noise or mismatch")
parser.add_argument('--mismatch', type=float, help='training masses or frictions')
parser.add_argument("--friction", default=False, action='store_true')
parser.add_argument("--max-steps", type=int, default=5000)
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)

if not args.no_compute:
    to_plot = []
    to_plot_std = []
    subfolder = "TestRobustenv" + str(args.env_name)
    if not os.path.isdir(assets_dir(subfolder)):
        os.makedirs(assets_dir(subfolder))

    torch.manual_seed(0)
    env = gym.make(args.env_name)
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
            for mass_mulL in args.mass_muls:
                env = gym.make(args.env_name, render_mode='rgb_array',
                                action_noise_mean=mass_mul,
                                action_noise_var=mass_mul)
                env.seed(0)
                means_avg = []
                stds_avg = []
                _, running_state = pickle.load(
                    open(os.path.join(assets_dir("env" + args.env_name + "noise_var0.0"),
                                      'expert_traj/' + args.env_name + "_expert_traj.p"), "rb"))
                for seed in args.seeds:
                    means = []
                    data_subfolder = "env" + args.env_name + "noise_varL" + str(args.mismatch) + "noise_varE0.0"
                    if not alg == "expert":
                        if os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                        'learned_models/{}_{}_best_{}.p'.format(
                                                            args.env_name + str(seed), alg, alpha))) > 0 and args.best:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_{}_best_{}.p'.format(
                                                                                 args.env_name + str(seed), alg,
                                                                                 alpha)), "rb"))
                        elif os.path.getsize(os.path.join(assets_dir(data_subfolder),
                                                          'learned_models/{}_{}_{}.p'.format(
                                                              args.env_name + str(seed), alg,
                                                              alpha))) > 0 and not args.best:
                            policy_net, _, _ = pickle.load(open(os.path.join(assets_dir(data_subfolder),
                                                                             'learned_models/{}_{}_{}.p'.format(
                                                                                 args.env_name + str(seed), alg,
                                                                                 alpha)), "rb"))

                        expert_flag = False
                    else:
                        policy_net, _, _ = pickle.load(
                            open(os.path.join(assets_dir("learned_models"), args.env_name + "_ppo.p"), "rb"))
                        expert_flag = True

                    if not running_state is None:
                        running_state.fix = True
                    else:
                        running_state = lambda x: x

                    for _ in range(10):
                        mean_reward = evaluate_loop(policy_net, running_state, expert_flag, env, is_disc_action, args)
                        means.append(mean_reward)

                    means_avg.append(np.mean(means))
                    stds_avg.append(np.std(means))

                to_append.append(np.mean(means_avg))
                to_append_std.append(np.mean(stds_avg))
                # to_append_std.append(np.std(means_avg))
            if alg == "expert" and expert_done:
                break
            if not alg == "expert":
                to_append = np.array(to_append)
                to_append_std = np.array(to_append_std)
                to_plot.append(to_append)
                to_plot_std.append(to_append_std)
            else:
                to_append = np.array(to_append)
                ones = np.ones_like(to_append)
                to_append = np.mean(to_append) * ones
                to_append_std = np.mean(to_append_std) * ones
                to_plot.append(to_append)
                to_plot_std.append(to_append_std)
            if alg == "expert":
                expert_done = True
        base_names = [alg_name for alg_name in ["expert"] if alg_name in args.alg]
        names = base_names + [alg_name + alpha for alpha in args.alpha for alg_name in args.alg]

        if not args.best:
            pickle.dump((to_plot, to_plot_std, names), open("../plot/" + subfolder + "/DataCompareAlphas"
                                                            + args.env_name + str(args.seeds) + args.alg[0] + ".p", 'wb'))
        else:
            pickle.dump((to_plot, to_plot_std, names), open("../plot/" + subfolder + "/DataCompareAlphas"
                                                            + args.env_name + str(args.seeds) + args.alg[0] + "best.p", 'wb'))
else:
    subfolder = "TestRobustenv" + str(args.env_name) + "friction" + str(args.friction)
    if not args.best:
        to_plot_load, to_plot_std_load, names_load = pickle.load(open("../plot/" + subfolder + "/DataCompareAlphas"
                                                                      + args.env_name + str(args.seeds) + ".p", 'rb'))
    else:
        to_plot_load, to_plot_std_load, names_load = pickle.load(open("../plot/" + subfolder + "/DataCompareAlphas"
                                                                      + args.env_name + str(args.seeds) + "best.p",
                                                                      'rb'))
    base_names = [alg_name for alg_name in ["expert"] if alg_name in args.alg]
    names = base_names + [alg_name + alpha for alpha in args.alpha for alg_name in args.alg]
    to_plot = [p for i, p in enumerate(to_plot_load) if names_load[i] in names]
    to_plot_std = [p for i, p in enumerate(to_plot_std_load) if names_load[i] in names]

plot.plot_lines_and_ranges(list_to_plot=to_plot,
                           list_sigmas=to_plot_std,
                           list_name=names,
                           axis_label=["Mass", "Total Reward"] if not args.friction else ["Friction", "Total Reward"],
                           folder="../plot/" + subfolder + "/",
                           title="RobustTest" + args.alg[0] + "CompareAlphas" + args.env_name + "best" + str(args.best) + str(
                               args.seeds) + "friction" + str(args.friction)+"mismatch" + str(args.mismatch),
                           x_axis=args.mass_muls)
