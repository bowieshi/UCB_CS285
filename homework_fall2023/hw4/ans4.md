# Homework 4

## 2 Analysis
pass

## 3 Model-Based Reinforcement Learning
### Problem 1
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_0_iter.yaml
```
#### experiment 1
num_layers: 1
hidden_size: 32
![itr_0_loss_curve.png](imgs%2Fitr_0_loss_curve.png)

#### experiment 2
num_layers: 4
hidden_size: 32
![itr_0_loss_curve2.png](imgs%2Fitr_0_loss_curve2.png)

#### experiment 3
num_layers: 1
hidden_size: 64
![itr_0_loss_curve3.png](imgs%2Fitr_0_loss_curve3.png)

### Problem 2
sripts:
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_1_iter.yaml```
```
first run:
```
Evaluating 20 rollouts...
Average eval return: -70.89758015261309
```
second run:
```
Evaluating 20 rollouts...
Average eval return: -50.63157355952895
```
third run:
```
Evaluating 20 rollouts...
Average eval return: -63.022171033950166
```

### Problem 3
#### Experiment 1
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/obstacles_multi_iter.yaml
```
![P3_E1_img1.png](imgs%2FP3_E1_img1.png)
![P3_E1_img2.png](imgs%2FP3_E1_img2.png)
#### Experiment 2
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_multi_iter.yaml
```
![P3_E2_img1.png](imgs%2FP3_E2_img1.png)
![P3_E2_img2.png](imgs%2FP3_E2_img2.png)
#### Experiment 3
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_multi_iter.yaml
```
![P3_E3_img1.png](imgs%2FP3_E3_img1.png)
![P3_E3_img2.png](imgs%2FP3_E3_img2.png)
### Problem 4
sripts:
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/reacher_ablation.yaml
```
Ablation study on `ensemble_size`.
- Orange line: `ensemble_size=1`
- Green line: `ensemble_size=3`
- black line: `ensemble_size=5`

![P4_E1_img1.png](imgs%2FP4_E1_img1.png)
![P4_E1_img2.png](imgs%2FP4_E1_img2.png)

Ablation study on `mpc_num_action_sequences`.
- Blue line: `mpc_num_action_sequences=500`
- Green line: `mpc_num_action_sequences=1000`
- Red line: `mpc_num_action_sequences=1500`

![P4_E2_img1.png](imgs%2FP4_E2_img1.png)
![P4_E2_img2.png](imgs%2FP4_E2_img2.png)

Ablation study on `mpc_horizon`.
- Yellow line: `mpc_horizon=5`
- Green line: `mpc_horizon=10`
- Purple line: `mpc_horizon=20`

![P4_E3_img1.png](imgs%2FP4_E3_img1.png)
![P4_E3_img2.png](imgs%2FP4_E3_img2.png)

### Problem 5
scripts:
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_cem.yaml
```
Comparison between `cem_iterations`. 
- Orange line: `cem_iterations=4`
- Purple line: `cem_iterations=2`

![P5_img1.png](imgs%2FP5_img1.png)
![P5_img2.png](imgs%2FP5_img2.png)

Comparison between `cem` and `random_shooting`. 
- Orange line: `cem_iterations=4`
- Purple line: `cem_iterations=2`
- Black line: `random_shooting`

![P5_img3.png](imgs%2FP5_img3.png)
![P5_img4.png](imgs%2FP5_img4.png)

### Problem 6
scripts:
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_mbpo.yaml --sac_config_file experiments/sac/halfcheetah_clipq.yaml
```
with different `mbpo_rollout_length`
- Blue line: `mbpo_rollout_length=0`
- Purple line: `mbpo_rollout_length=1`
- Green line: `mbpo_rollout_length=10`

![P6_img2.png](imgs%2FP6_img2.png)
![P6_img3.png](imgs%2FP6_img3.png)
![P6_img1.png](imgs%2FP6_img1.png)