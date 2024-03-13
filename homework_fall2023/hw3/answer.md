# Homework 3

## Deep Q-learning

### Deliver 1: DQN on CartPole-v1

shell script:
```shell
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/cartpole.yaml
```
learning rate is 0.001:
![DQN-CartPole-v1.png](imgs%2FDQN-CartPole-v1.png)
two lines have the same configure but different run.
![DQN-CartPole-v1lr.png](imgs%2FDQN-CartPole-v1lr.png)
learning rate is 0.05 (by modifying yaml file):
![DQN-CartPole-v1-modifyLR1.png](imgs%2FDQN-CartPole-v1-modifyLR1.png)
![DQN-CartPole-v1-modifyLR2.png](imgs%2FDQN-CartPole-v1-modifyLR2.png)
![DQN-CartPole-v1-modifyLR3.png](imgs%2FDQN-CartPole-v1-modifyLR3.png)
#### Analysis
a) Effect on the predicted Q-values:

The predicted Q-values tend to become more unstable and fluctuate more significantly during training. A higher learning rate means that the neural network weights are updated more aggressively based on the calculated gradients. This can cause the Q-values to change drastically from one iteration to the next, resulting in larger fluctuations.

b) Effect on the critic error:

The critic error, also known as the temporal difference (TD) error or the Bellman error, measures the difference between the predicted Q-value and the target Q-value calculated using the Bellman equation. With a higher learning rate, the critic error is likely to be more volatile and may take longer to converge.

When the learning rate is too high, the neural network weights can overshoot the optimal values, leading to larger fluctuations in the critic error. This can cause the training to become unstable, and the agent may struggle to learn an optimal policy efficiently.

### Deliver 2: DQN on LunarLander-v2

shell script:
```shell
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander.yaml --seed 3
```
![DQN-LunarLander.png](imgs%2FDQN-LunarLander.png)
### Deliver 3: 

shell script:
```shell
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 1
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 2
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/lunarlander_doubleq.yaml --seed 3
```
![DQN-LunarLander-doubleq.png](imgs%2FDQN-LunarLander-doubleq.png)

### Deliver 4:

shell script:
```shell
python cs285/scripts/run_hw3_dqn.py -cfg experiments/dqn/mspacman.yaml
```
![DQN-MsPacman.png](imgs%2FDQN-MsPacman.png)


## Soft-Actor-Critic learning

test first section(without any trick):
```shell
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/sanity_pendulum.yaml
```

### 3.1.3 Actor with REINFORCE

#### Testing this section

Train an agent on InvertedPendulum-v4 using sanity_invertedpendulum_reinforce.yaml. You should achieve reward close to 1000, which corresponds to staying upright for all time steps.

scripts:
```shell
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/sanity_invertedpendulum_reinforce.yaml
```
![SAC_test_313.png](imgs%2FSAC_test_313.png)

#### Deliverable 1

Train an agent on HalfCheetah-v4 using the provided config (halfcheetah_reinforce1.yaml). Note
that this configuration uses only one sampled action per training example.

scripts:
```shell
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce1.yaml
```

[To be run]

#### Deliverable 2

Train another agent with halfcheetah_reinforce_10.yaml. This configuration takes many samples
from the actor for computing the REINFORCE gradient (we’ll call this REINFORCE-10, and the singlesample version REINFORCE-1). Plot the results (evaluation return over time) on the same axes as the
single-sample REINFORCE. Compare and explain your results.

scripts:
```shell
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reinforce_10.yaml
```

[To be run]

### 3.1.4 Actor with REPARAMETRIZE

#### Testing this section

Make sure you can solve InvertedPendulum-v4 (use sanity_invertedpendulum_reparametrize.yaml) and achieve similar reward to the REINFORCE case.

```shell
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/sanity_invertedpendulum_reparametrize.yaml
```

![SAC_test_314.png](imgs%2FSAC_test_314.png)

#### Deliverable 1

Train (once again) on HalfCheetah-v4 with halfcheetah_reparametrize.yaml. Plot results for all
three gradient estimators (REINFORCE-1, REINFORCE-10 samples, and REPARAMETRIZE) on the
same set of axes, with number of environment steps on the x-axis and evaluation return on the y-axis.

scripts:
```shell
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/halfcheetah_reparametrize.yaml
```

[To be run]

#### Deliverable 2

Train an agent for the Humanoid-v4 environment with humanoid_sac.yaml and plot results

scripts:
```shell
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/humanoid_sac.yaml
```

[To be run]

### 3.1.5 Stabilizing Target Values

#### Deliverable 1

Run single-Q, double-Q, and clipped double-Q on Hopper-v4 using the corresponding configuration files.
Which one works best? Plot the logged eval_return from each of them as well as q_values. Discuss
how these results relate to overestimation bias.

```shell
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper_doubleq.yaml
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/hopper_clipq.yaml
```

[To be run]

#### Deliverable 2

Pick the best configuration (single-Q/double-Q/clipped double-Q, or REDQ if you implement it) and
run it on Humanoid-v4 using humanoid.yaml (edit the config to use the best option). You can truncate it
after 500K environment steps. If you got results from the humanoid environment in the last homework,
plot them together with environment steps on the x-axis and evaluation return on the y-axis. Otherwise,
we will provide a humanoid log file that you can use for comparison. How do the off-policy and on-policy
algorithms compare in terms of sample efficiency? Note: if you’d like to run training to completion (5M
steps), you should get a proper, walking humanoid! You can run with videos enabled by using -nvid 1.
If you run with videos, you can strip videos from the logs for submission with this script.

```shell
python cs285/scripts/run_hw3_sac.py -cfg experiments/sac/humanoid.yaml
```

[To be run]
