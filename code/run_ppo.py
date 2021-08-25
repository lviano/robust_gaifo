import argparse
import os
import numpy as np
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='path of pre-trained model')
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
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=1, metavar='N',
                    help="interval between saving model")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N')
parser.add_argument('--noiseE', type=float, default=None, metavar='G')
parser.add_argument('--grid-type', type=int, default=None, metavar='N')
parser.add_argument('--mass-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot masses")
parser.add_argument('--len-mul', type=float, default=1.0, metavar='G',
                    help="Multiplier for CartPole and Acrobot lengths")
parser.add_argument('--friction', default=False, action='store_true')
parser.add_argument("--script_name", default='submit.sh')
parser.add_argument("--logs_folder", default='./logs')
parser.add_argument("--job_name", default='')
args = parser.parse_args()

if not os.path.isfile(args.script_name):
    with open(args.script_name, 'w') as file:
        file.write(f'''#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --mem-per-cpu=1000

./staskfarm ${{1}}\n''')
folder = f'{args.logs_folder}/plot/'
if not args.friction:
    path = f'env_name_{args.env_name}_mass_mul_{args.mass_mul}_len_mul_{args.len_mul}/'
else:
    path = f'env_name_{args.env_name}_friction_mul_{args.mass_mul}/'
file = f'examples/ppo_gym.py'
if not os.path.isdir(f'{folder}/{path}'):
    os.makedirs(f'{folder}/{path}')
command = f'python {file} --env-name {args.env_name}  --save-model-interval {args.save_model_interval} ' \
          f'--log-interval {args.log_interval} --num-threads {args.num_threads} --learning-rate {args.learning_rate} ' \
          f'--max-iter-num {args.max_iter_num} --mass-mul {args.mass_mul} --len-mul {args.len_mul} '
if args.friction:
    command += '--friction '
if args.env_name == "gridworld-v0" or args.env_name == "ContinuousGridworld-v0" or args.env_name == "GaussianGridworld-v0" :
    command = f'{command} --noiseE {args.noiseE}  --grid-type {args.grid_type}'

experiment_path = f'{folder}/{path}/command.txt'

with open(experiment_path, 'w') as file:
    file.write(f'{command}\n')

print(command)

if not args.job_name:
    job_name = path
else:
    job_name = args.job_name

os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
