# Homework 4

## 2 Analysis

### Problem 2.1

Alternative Simulation Lemma

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

### Problem 5
scripts:
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_cem.yaml
```

### Problem 6
scripts:
```bash
python cs285/scripts/run_hw4.py -cfg experiments/mpc/halfcheetah_mbpo.yaml --sac_config_file experiments/sac/halfcheetah_clipq.yaml
```