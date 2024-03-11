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

