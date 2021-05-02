import argparse
import os
import numpy as np
import ast

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
parser.add_argument("--script_name", default='submit.sh')
parser.add_argument("--logs_folder", default='./logs')
parser.add_argument("--job_name", default='')
parser.add_argument("--exp-type", type=str, default="mismatch")
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
if args.env_name == "gridworld-v0":
    folder = f'{args.logs_folder}/plot/'

    path = f'env_name_{args.env_name}/grid-type_{args.grid_type}'
    file = f'analysis/compare_policy_performance.py'
    if not os.path.isdir(f'{folder}/{path}'):
        os.makedirs(f'{folder}/{path}')
    command = f'python {file} --env-name {args.env_name} --noiseL {args.noiseL}  --grid-type {args.grid_type} --alg '
    for alg in args.alg:
        command = f'{command} {alg} '
    command = f'{command} --seeds '
    for seed in args.seeds:
        command = f'{command} {seed} '
    command = f'{command} --noiseE '
    for noiseE in args.noiseE:
        command = f'{command} {noiseE} '
    command = f'{command} --alpha '
    for alpha in args.alpha:
        command = f'{command} {alpha} '

    if args.no_compute:
        command = f'{command} --no-compute'
    if args.vi_expert:
        command = f'{command} --vi-expert'
    if args.best:
        command = f'{command} --best'

    experiment_path = f'{folder}/{path}/command.txt'

    with open(experiment_path, 'w') as file:
        file.write(f'{command}\n')

    print(command)

    if not args.job_name:
        job_name = path
    else:
        job_name = args.job_name

    os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')

if args.env_name == "Ant-v2" or args.env_name == "HalfCheetah-v2":
    folder = f'{args.logs_folder}/plot/'

    path = f'env_name_{args.env_name}_best_{args.best}_{args.exp_type}/'
    file = f'analysis/compare_policy_performance.py'
    if not os.path.isdir(f'{folder}/{path}'):
        os.makedirs(f'{folder}/{path}')
    command = f'python {file} --env-name {args.env_name} --alg '
    for alg in args.alg:
        command = f'{command} {alg} '
    command = f'{command} --seeds '
    for seed in args.seeds:
        command = f'{command} {seed} '
    command = f'{command} --noiseE '
    for noiseE in args.noiseE:
        command = f'{command} {noiseE} '
    command = f'{command} --alpha '
    for alpha in args.alpha:
        command = f'{command} {alpha} '

    command = f'{command} --exp-type {args.exp_type} '

    if args.no_compute:
        command = f'{command} --no-compute'

    if args.best:
        command = f'{command} --best'

    experiment_path = f'{folder}/{path}/command.txt'

    with open(experiment_path, 'w') as file:
        file.write(f'{command}\n')

    print(command)

    if not args.job_name:
        job_name = path
    else:
        job_name = args.job_name

    os.system(f'sbatch --job-name={job_name} {args.script_name} {experiment_path}')
