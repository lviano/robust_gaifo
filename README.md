***Implementation of robust GAIL under transition dynamics mismatch***

*** Installation ***

You can replicate the virtual environment we used in our experiments with the file `requirements.txt`

*** Run the experiments ***
The commands we present here must be executed from the folder `code`.


The first step is to train an expert in the nominal environment. We offer the possibility to do this using PPO on any environment that offers 
a GYM interface. Running


```
python run_ppo.py --env-name <environment name> --max-iter-num 3000 --save-model-interval 500
```


The model is saved in the folder `code/assets/learned_model` by default. However you can change the destination changing the variable `subfolder` in `examples/ppo_gym.py`.
Then you can save trajectories with the command:


```
python gail/save_expert_traj.py --env-name <environment name> --max-expert-state-num 1000 --model-path <path-to-the-PPO-model>
```


In case of gridworld environments, you can save the trajectories computed by Value Iteration with the command:


```python gail/value_iteration_gym.py --max-expert-state-num 3000 --noiseE 0.0 --grid-type 1```


The trajectories are saved by default in the folder `assets/env<environment-name>/expert_traj`. However, you can change that name editing the file `gail/save_expert_traj.py`


Run Robust GAIFO with Slurm:


``` python run_experiment.py --env-name <environment-name> --algo gaifo --learning-rate 1e-4 --alpha 1.0 0.999 0.99 0.98 0.97 0.96 0.95 0.9 --num-threads 1 --min-batch-size 3000 --max-iter-num 1000 --log-interval 1 --save-model-interval 1 --expert-traj-path <path-to-trajectories> --seed 0 1 2```


As a concrete example:


```
 python run_experiment.py --env-name InvertedDoublePendulum-v2 --algo gaifo --learning-rate 1e-4 
--alpha 1.0 0.999 0.99 0.98 0.97 0.96 0.95 0.9 --num-threads 1 --min-batch-size 3000 --max-iter-num 500 
--log-interval 1 --save-model-interval 1 --mass-mulL 0.5 0.75 1.0 1.25 1.5 2.0  --mass-mulE 1.0 
--expert-traj-path assets/envInvertedDoublePendulum-v2mass1.0/expert_traj/InvertedDoublePendulum-v2_state_only_expert_traj.p 
--seed 2 3 4 --reward-type positive --exp-type friction
```

From the concrete example we notice that we can pass different "mismatch" values as `--mass-mulL`. 
In the experiments we did, these values correspond to multipliers for the mass or friction of the learner.

In order to incorporate a mismatch relevant to your experiment please edit the file `gail/algo_gym.py` similarly to how we change the mass of the MuJoCo agents.

Please, edit also `run_experiment.py` to indicate a proper saving path, if the one for MuJoCo is fine for your situation, just add the environment name to the if condition at line `90`.


Similarly, you can run the AIRL baseline as:

Run AIRL:
```
 python run_experiment.py --env-name InvertedDoublePendulum-v2 --algo airl --learning-rate 1e-4 
--alpha 1.0 --num-threads 1 --min-batch-size 3000 --max-iter-num 500 
--log-interval 1 --save-model-interval 1 --mass-mulL 0.5 0.75 1.0 1.25 1.5 2.0  --mass-mulE 1.0 
--expert-traj-path assets/envInvertedDoublePendulum-v2mass1.0/expert_traj/InvertedDoublePendulum-v2_state_only_expert_traj.p 
--seed 2 3 4 --reward-type positive --exp-type friction
```

Please notice that when you use gaifo or airl as an algorithm you must passe state_only trajectories that are saved by default under the name `<environment-name>_state_only_expert_traj.p`.

Finally, notice the variable `reward-type`. It is a well known issue in imitation learning that for some environment -log(D) works better than log(1-D) as a reward. 
Select `negative` to use the former or `positive` for the latter.

Evaluation:

In the work, we are using 2 different type of evaluations. 

***Learning under mismatch***

To evaluate the expert and gaifo for different alphas

```
python analysis/compare_mujoco_performance.py --env-name <environment-name> --alg expert gaifo --alpha 1.0 0.999 0.99 0.98 0.97 0.96 0.95 0.9 --mass-muls 0.5 0.75 1.0 1.5 2.0 --seed 2 3 4 --friction
``` 

the argument `--friction` can be removed to evaluate the `mass` mismatch rather than the `friction` one.

while for evaluate airl:

```
python analysis/compare_mujoco_performance.py --env-name <environment-name> --alg airl --alpha 1.0 0.999 0.99 0.98 0.97 0.96 0.95 0.9 --mass-muls 0.5 0.75 1.0 1.5 2.0 --seed 2 3 4 --friction
```

***Evaluating the robustness of learned *** 

To evaluate the expert and gaifo for different alphas

```
python analysis/compare_mujoco_robustness.py --env-name <environment-name> --alg gaifo --alpha 1.0 0.999 0.99 0.98 0.97 0.96 0.95 0.9 --mass-muls 0.5 0.75 1.0 1.5 2.0 --seed 0 1 2 --mismatch 1.5 --friction
``` 

while for evaluate airl:

```
python analysis/compare_mujoco_robustness.py --env-name <environment-name> --alg airl --alpha 1.0 0.999 0.99 0.98 0.97 0.96 0.95 0.9 --mass-muls 0.5 0.75 1.0 1.5 2.0 --seed 2 3 4 --friction
```

The argument `mismatch` denotes the mismatch used for learning. While the argument `mass-muls` expects the range of masses to evaluate at testing time. 


