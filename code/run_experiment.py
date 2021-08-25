import argparse
import os
import numpy as np
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--algo', default="gail", metavar='G',
                    help='algorithm to run. Options are gail or robust_gail')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument("--alpha", nargs='+', default=["0"])
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument("--seed", nargs='+', default=["0"])
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument("--noiseL", nargs='+', default=["0"])
parser.add_argument("--noiseE", type=float, default = 0.0)
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mulL', nargs='+', default=["0"])
parser.add_argument('--len-mulL', nargs='+', default=["0"])
parser.add_argument('--mass-mulE', type=float, default = 1.0)
parser.add_argument('--len-mulE', type=float, default = 1.0)
parser.add_argument('--scheduler-lr', action='store_true', default=False,
                    help='Use discriminator lr scheduler')
parser.add_argument('--reward-type', type=str, default="negative")
parser.add_argument("--exp-type", type=str, default="mismatch")
parser.add_argument("--script_name", default='submit.sh')
parser.add_argument("--logs_folder", default='./logs')
parser.add_argument("--job_name", default='')
args = parser.parse_args()

# If submit script does not exist, create i
if not os.path.isfile(args.script_name):
    with open(args.script_name, 'w') as file:
        file.write(f'''#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=1000

./staskfarm ${{1}}\n''')
if args.env_name == "GaussianGridworld-v0":
    for seed in args.seed:
        for alpha in args.alpha:
            for noiseL in args.noiseL:
                folder = f'{args.logs_folder}/gw'

                path = f'env_name_{args.env_name}/' \
                       f'grid_type_{args.grid_type}/alg{args.algo}/alpha_{alpha}/' \
                       f'lr_{args.learning_rate}/' \
                       f'noiseE_{args.noiseE}/' \
                       f'noiseL_{noiseL}/' \
                       f'seed_{seed}'


                if not os.path.isdir(f'{folder}/{path}'):
                    os.makedirs(f'{folder}/{path}')

                file = f'gail/algo_gym.py'
                command = f'python {file} --env-name {args.env_name} --alg {args.algo} ' \
                          f'--grid-type {args.grid_type} ' \
                          f'--expert-traj-path {args.expert_traj_path} ' \
                          f'--num-threads {args.num_threads} ' \
                          f'--log-interval {args.log_interval} ' \
                          f'--save-model-interval {args.save_model_interval} ' \
                          f'--max-iter-num {args.max_iter_num} ' \
                          f'--learning-rate {args.learning_rate} --alpha {alpha} ' \
                          f'--seed {seed} ' \
                          f'--noiseE {args.noiseE} ' \
                          f'--noiseL {noiseL}'

                experiment_path = f'{folder}/{path}/command.txt'

                with open(experiment_path, 'w') as file:
                    file.write(f'{command}\n')

                print(command)

                if not args.job_name:
                    job_name = path
                else:
                    job_name = args.job_name

                os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
elif args.env_name == "gridworld-v0" :
    for seed in args.seed:
        for alpha in args.alpha:
            for noiseL in args.noiseL:
                folder = f'{args.logs_folder}/gw'

                path = f'env_name_{args.env_name}/grid_type_{args.grid_type}/alg{args.algo}_alpha_{alpha}' \
                       f'/lr_{args.learning_rate}/noiseE_{args.noiseE}' \
                       f'/noiseL_{noiseL}/seed_{seed}'


                if not os.path.isdir(f'{folder}/{path}'):
                    os.makedirs(f'{folder}/{path}')


                    file = f'gail/algo_gym.py'
                    command = f'python {file} --env-name {args.env_name} --alg {args.algo} --grid-type {args.grid_type} ' \
                              f'--expert-traj-path {args.expert_traj_path} --num-threads {args.num_threads} ' \
                              f'--log-interval {args.log_interval} --save-model-interval {args.save_model_interval} ' \
                              f'--min-batch-size {args.min_batch_size} --max-iter-num {args.max_iter_num} ' \
                              f'--learning-rate {args.learning_rate} --seed {seed} --noiseE {args.noiseE} ' \
                              f'--noiseL {noiseL} --alpha {alpha}'

                experiment_path = f'{folder}/{path}/command.txt'

                with open(experiment_path, 'w') as file:
                    file.write(f'{command}\n')

                print(command)

                if not args.job_name:
                    job_name = path
                else:
                    job_name = args.job_name

                os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
elif args.env_name == "HalfCheetah-v2" or args.env_name == "Ant-v2" or args.env_name == "Walker2d-v2" or args.env_name == "Hopper-v2" or args.env_name=="Swimmer-v2" or args.env_name == "InvertedDoublePendulum-v2" or args.env_name == "InvertedPendulum-v2":
    for massL in args.mass_mulL:
            for seed in args.seed:
                    for alpha in args.alpha:
                        folder = f'{args.logs_folder}/mujoco'

                        path = f'env_name_{args.env_name}/alg_{args.algo}/' \
                           f'_alpha_{alpha}/lr_{args.learning_rate}/noiseE_{args.noiseE}/exp_type_{args.exp_type}/massE_{args.mass_mulE}' \
                           f'/massL_{massL}/seed_{seed}'


                        if not os.path.isdir(f'{folder}/{path}'):
                            os.makedirs(f'{folder}/{path}')

                        file = f'gail/algo_gym.py'
                        command = f'python {file} --env-name {args.env_name} --alg {args.algo} --expert-traj-path {args.expert_traj_path} ' \
                              f'--num-threads {args.num_threads} --log-interval {args.log_interval} ' \
                              f'--save-model-interval {args.save_model_interval} ' \
                              f'--min-batch-size {args.min_batch_size} --max-iter-num {args.max_iter_num} ' \
                              f'--learning-rate {args.learning_rate} --seed {seed}' \
                              f' --noiseE {args.noiseE} --alpha {alpha} --mass-mulE {args.mass_mulE} ' \
                              f'--mass-mulL {massL}  --exp-type {args.exp_type} --reward-type {args.reward_type}'

                        experiment_path = f'{folder}/{path}/command.txt'

                        with open(experiment_path, 'w') as file:
                            file.write(f'{command}\n')

                        print(command)

                        if not args.job_name:
                            job_name = path
                        else:
                            job_name = args.job_name

                        os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
elif args.env_name == "Acrobot-v1" or args.env_name == "CartPole-v1":
    for seed in args.seed:
        for alpha in args.alpha:
            for mass_mulL in args.mass_mulL:
                for len_mulL in args.len_mulL:
                    folder = f'{args.logs_folder}'

                    path = f'env_name_{args.env_name}/grid_type_{args.grid_type}' \
                           f'/alpha_{alpha}/lr_{args.learning_rate}/exp_traj_{args.expert_traj_path}' \
                           f'/mass_mulL_{mass_mulL}/len_mulL_{len_mulL}/seed_{seed}'


                    if not os.path.isdir(f'{folder}/{path}'):
                        os.makedirs(f'{folder}/{path}')

                    file = f'gail/algo_gym.py'
                    command = f'python {file} --env-name {args.env_name} --alg {args.algo} ' \
                              f'--expert-traj-path {args.expert_traj_path} --num-threads {args.num_threads} ' \
                              f'--log-interval {args.log_interval} --save-model-interval {args.save_model_interval} ' \
                              f'--min-batch-size {args.min_batch_size} --max-iter-num {args.max_iter_num} ' \
                              f'--learning-rate {args.learning_rate} --seed {seed} ' \
                              f'--mass-mulE {args.mass_mulE} --mass-mulL {mass_mulL} --len-mulE {args.len_mulE} ' \
                              f' --len-mulL {len_mulL} --alpha {alpha}'

                    experiment_path = f'{folder}/{path}/command.txt'

                    with open(experiment_path, 'w') as file:
                        file.write(f'{command}\n')

                    print(command)

                    if not args.job_name:
                        job_name = path
                    else:
                        job_name = args.job_name

                    os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
elif args.env_name == "gym_reach:reachNoisy-v0":
    for massL in args.mass_mulL:
            for seed in args.seed:
                    for alpha in args.alpha:
                        folder = f'{args.logs_folder}/gym_reach'

                        path = f'env_name_{args.env_name}/alg_{args.algo}/' \
                           f'_alpha_{alpha}/lr_{args.learning_rate}/noiseE_{args.noiseE}/exp_type_{args.exp_type}/massE_{args.mass_mulE}' \
                           f'/massL_{massL}/seed_{seed}'


                        if not os.path.isdir(f'{folder}/{path}'):
                            os.makedirs(f'{folder}/{path}')

                        file = f'gail/algo_gym.py'
                        command = f'python {file} --env-name {args.env_name} --alg {args.algo} --expert-traj-path {args.expert_traj_path} ' \
                              f'--num-threads {args.num_threads} --log-interval {args.log_interval} ' \
                              f'--save-model-interval {args.save_model_interval} ' \
                              f'--min-batch-size {args.min_batch_size} --max-iter-num {args.max_iter_num} ' \
                              f'--learning-rate {args.learning_rate} --seed {seed}' \
                              f' --noiseE {args.noiseE} --alpha {alpha} --mass-mulE {args.mass_mulE} ' \
                              f'--mass-mulL {massL}  --exp-type {args.exp_type} --reward-type {args.reward_type}'

                        experiment_path = f'{folder}/{path}/command.txt'

                        with open(experiment_path, 'w') as file:
                            file.write(f'{command}\n')

                        print(command)

                        if not args.job_name:
                            job_name = path
                        else:
                            job_name = args.job_name

                        os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
