# homework 1
## 1 Analysis
### Problem 1
$$
assume\; \pi_{\theta}(s_t\ne\pi^{*}(s_t)|s_t)\leq \epsilon_{s_t}\\
p_{\theta}(s_t)=(1-\epsilon)^t p_{train}(s_t)+(1-(1-\epsilon)^t)p_{mistake}(s_t)\\
|p_{\theta}(s_t)-p_{train}(s_t)|=(1-(1-\epsilon)^t)|p_{mistake}(s_t)-p_{train}(s_t)|
\\ \leq 2(1-(1-\epsilon)^t)\leq 2t\epsilon_{s_t}
$$
From the inequailty given, we have
$$
\mathbb{E}_{p_{\pi^{*}(s)}}\pi_{\theta}(a\ne\pi^{*}(s)|s)=\frac{1}{T}\sum_{t=1}^T\mathbb{E}_{p_{\pi^*(s_t)}}\pi_{\theta}(a_t\ne\pi^*(s_t)|s_t)\leq \epsilon
$$

$$
\sum_{s_t}|p_{\pi_{\theta}}(s_t)-p_{\pi^{*}}(s_t)|\leq
$$

## 2 Editiing Cod

finish code editing

## 3 Behavioral Cloning
```shell
export PYTHONPATH="${PYTHONPATH}:/home/astar/bowieshi/inno1_remote/UCB_CS285/homework_fall2023/hw1"```
export MUJOCO_GL=egl
```
#### Ant
```shell
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1
```
#### HalfCheetah
```shell
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
--env_name HalfCheetah-v4 --exp_name bc_HalfCheetah --n_iter 1 \
--expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl \
--video_log_freq -1
```
#### Hopper
```shell
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Hopper.pkl \
--env_name Hopper-v4 --exp_name bc_Hopper --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
--video_log_freq -1
```
#### Walker2d
```shell
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 --exp_name bc_Walker2d --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
--video_log_freq -1
```
### Result
#### Ant
```text
Eval_StdReturn : 0.0
Eval_MaxReturn : 659.323486328125
Eval_MinReturn : 659.323486328125
Eval_AverageEpLen : 1000.0
Train_AverageReturn : 4681.891673935816
Train_StdReturn : 30.70862278765526
Train_MaxReturn : 4712.600296723471
Train_MinReturn : 4651.18305114816
Train_AverageEpLen : 1000.0
Training Loss : 0.03436639532446861
Train_EnvstepsSoFar : 0
TimeSinceStart : 13.550403118133545
Initial_DataCollection_AverageReturn : 4681.891673935816
```
#### HalfCheetah
```text
Eval_AverageReturn : 3180.66845703125
Eval_StdReturn : 0.0
Eval_MaxReturn : 3180.66845703125
Eval_MinReturn : 3180.66845703125
Eval_AverageEpLen : 1000.0
Train_AverageReturn : 4034.7999834965067
Train_StdReturn : 32.8677631311341
Train_MaxReturn : 4067.6677466276406
Train_MinReturn : 4001.9322203653724
Train_AverageEpLen : 1000.0
Training Loss : 0.04342725872993469
Train_EnvstepsSoFar : 0
TimeSinceStart : 25.803037643432617
Initial_DataCollection_AverageReturn : 4034.7999834965067
```
#### Hopper
```text
Collecting data for eval...
Eval_AverageReturn : 809.261474609375
Eval_StdReturn : 306.9176940917969
Eval_MaxReturn : 1061.2845458984375
Eval_MinReturn : 297.77008056640625
Eval_AverageEpLen : 250.5
Train_AverageReturn : 3717.5129936182307
Train_StdReturn : 0.3530361779417035
Train_MaxReturn : 3717.8660297961724
Train_MinReturn : 3717.159957440289
Train_AverageEpLen : 1000.0
Training Loss : 0.04094276204705238
Train_EnvstepsSoFar : 0
TimeSinceStart : 15.9848792552948
Initial_DataCollection_AverageReturn : 3717.5129936182307
```
#### Walker2d
```text
Collecting data for eval...
Eval_AverageReturn : 924.3436279296875
Eval_StdReturn : 848.614990234375
Eval_MaxReturn : 2381.55419921875
Eval_MinReturn : 261.83355712890625
Eval_AverageEpLen : 255.75
Train_AverageReturn : 5383.310325177668
Train_StdReturn : 54.15251563871789
Train_MaxReturn : 5437.462840816386
Train_MinReturn : 5329.1578095389505
Train_AverageEpLen : 1000.0
Training Loss : 0.05407339707016945
Train_EnvstepsSoFar : 0
TimeSinceStart : 6.776831865310669
Initial_DataCollection_AverageReturn : 5383.310325177668
```
![P3img1.png](imgs%2FP3img1.png)
### Comparation between hyperparameters
#### raw
```shell
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 --exp_name bc_Walker2d --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl
```
```text
Collecting data for eval...
Eval_AverageReturn : 924.3436279296875
Eval_StdReturn : 848.614990234375
Eval_MaxReturn : 2381.55419921875
Eval_MinReturn : 261.83355712890625
Eval_AverageEpLen : 255.75
Train_AverageReturn : 5383.310325177668
Train_StdReturn : 54.15251563871789
Train_MaxReturn : 5437.462840816386
Train_MinReturn : 5329.1578095389505
Train_AverageEpLen : 1000.0
Training Loss : 0.05407339707016945
Train_EnvstepsSoFar : 0
TimeSinceStart : 6.776831865310669
Initial_DataCollection_AverageReturn : 5383.310325177668
```
#### The amount of training steps
```shell
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 --exp_name bc_Walker2d --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
--num_agent_train_steps_per_iter 5000
```
```text
Collecting data for eval...
Eval_AverageReturn : 5321.7802734375
Eval_StdReturn : 0.0
Eval_MaxReturn : 5321.7802734375
Eval_MinReturn : 5321.7802734375
Eval_AverageEpLen : 1000.0
Train_AverageReturn : 5383.310325177668
Train_StdReturn : 54.15251563871789
Train_MaxReturn : 5437.462840816386
Train_MinReturn : 5329.1578095389505
Train_AverageEpLen : 1000.0
Training Loss : 0.009046615101397038
Train_EnvstepsSoFar : 0
TimeSinceStart : 34.9678316116333
Initial_DataCollection_AverageReturn : 5383.310325177668
```

#### the amount of expert data provided
```shell
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Walker2d.pkl \
--env_name Walker2d-v4 --exp_name bc_Walker2d --n_iter 1 \
--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
--batch 10000
```
```text
Collecting data for eval...
Eval_AverageReturn : 924.3436279296875
Eval_StdReturn : 848.614990234375
Eval_MaxReturn : 2381.55419921875
Eval_MinReturn : 261.83355712890625
Eval_AverageEpLen : 255.75
Train_AverageReturn : 5383.310325177668
Train_StdReturn : 54.15251563871789
Train_MaxReturn : 5437.462840816386
Train_MinReturn : 5329.1578095389505
Train_AverageEpLen : 1000.0
Training Loss : 0.05407339707016945
Train_EnvstepsSoFar : 0
TimeSinceStart : 6.752086162567139
Initial_DataCollection_AverageReturn : 5383.310325177668
```
The result shows that `batch_size=1000` expert data is enought to train a good policy.

However, the training does not converge when `num_agent_train_steps_per_iter=1000`.

When we increase the training steps, the policy improves a lot.

Even adding more expert data, the policy does not improve due to insufficient steps for convergence.
## 4 DAGGER
```shell
python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
--video_log_freq -1
```
![P4img1.png](imgs%2FP4img1.png)
![P4img2.png](imgs%2FP4img2.png)