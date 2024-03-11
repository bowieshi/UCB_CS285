# Section 3: Policy Gradient

## Graph 1 - The small batch experiments
shell scripts:
```shell
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name cartpole_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -na --exp_name cartpole_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -na --exp_name cartpole_rtg_na
```
![sec3graph1.png](imgs%2Fsec3graph1.png)
## Graph 2 - The large batch experiments
shell scripts
```shell
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 --exp_name cartpole_lb
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg --exp_name cartpole_lb_rtg
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -na --exp_name cartpole_lb_na
python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 100 -b 4000 -rtg -na --exp_name cartpole_lb_rtg_na
```
![sec3graph2.png](imgs%2Fsec3graph2.png)

## Analysis
– Which value estimator has better performance without advantage normalization: the trajectory-centric one, or the one using reward-to-go?

**reward-to-go is better**

– Did advantage normalization help?

**Yes, help converge faster and reduce AvgReturn variance.**

– Did the batch size make an impact?

**Yes, converge faster. But the variance is higher.**

# Section 4: Using a Neural Network Baseline

## Graph 1 - learning curve for the baseline loss
shell script
```shell
# No baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
    -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
    --exp_name cheetah

# Baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
    -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
    --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
```
![sec4graph1.png](imgs%2Fsec4graph1.png)
## Graph 2 - learning curve for the eval return
shell script
```shell
# No baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
    -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
    --exp_name cheetah

# Baseline
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
    -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
    --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline
```
![sec4graph2.png](imgs%2Fsec4graph2.png)

## Graph 3

### decreased number of baseline gradient steps (-bgs)    
shell scripts
```shell
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
    -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
    --use_baseline -blr 0.01 -bgs 2 --exp_name cheetah_baseline_decreased_bgs
```
### decreased number of baseline learning rate (-blr)
shell scripts
```shell
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
    -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 \
    --use_baseline -blr 0.001 -bgs 5 --exp_name cheetah_baseline_decreased_blr
```
### normalized advantage added and save video
shell scripts
```shell
python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 \
    -n 100 -b 5000 -rtg --discount 0.95 -lr 0.01 -na\
    --use_baseline -blr 0.01 -bgs 5 --exp_name cheetah_baseline_na_video
```
![sec4graph4.png](imgs%2Fsec4graph4.png)
![sec4graph3.png](imgs%2Fsec4graph3.png)
### Analysis
decreased number of baseline gradient steps (-bgs) and/or baseline
learning rate (-blr) will cause:
1. baseline learning much slower, but they can converge to almost same level at last;
2. performance of policy worser.

If use normalized advantage:
the policy perform the best.

# Section 5: Implementing Generalized Advantage Estimation
## Graph 1
shell scripts
```shell
python cs285/scripts/run_hw2.py \
    --env_name LunarLander-v2 --ep_len 1000 \
    --discount 0.99 -n 300 -l 3 -s 128 -b 2000 -lr 0.001 \
    --use_reward_to_go --use_baseline --gae_lambda 1 \
    --exp_name lunar_lander_lambda1
```
![sec5graph1.png](imgs%2Fsec5graph1.png)
![sec5graph2.png](imgs%2Fsec5graph2.png)

larger $\lambda$ has better performance.
with lambda being 0, we never consider future rewards, we only use present reward when you choose a_t in s_t to update policy.
with lambda being 1, we regard present reward and future rewards with the same importance.

# Section 6: Hyperparameters and Sample Efficiency



# Section 7: Humanoid

shell script
```shell
nohup python cs285/scripts/run_hw2.py \
    --env_name Humanoid-v4 --ep_len 1000 \
    --discount 0.99 -n 1000 -l 3 -s 256 -b 50000 -lr 0.001 \
    --baseline_gradient_steps 50 \
    -na --use_reward_to_go --use_baseline --gae_lambda 0.97 \
    --exp_name humanoid --video_log_freq 5 > train_log &
```
# Section 8: Analysis
## (a)
$$
\begin{align*}
    J(\theta)=E_{\pi_{\theta}}R(\tau)
    \\=\sum_{t=1}^{\infin}\theta^{t}
    \\\frac{d}{d\theta}J(\theta)=\frac{d}{d\theta}\sum_{t=1}^{\infin}\theta^{t}
    \\=\frac{d}{d\theta}\frac{\theta}{1-\theta}
    \\=\frac{1}{(1-\theta)^2}
\end{align*}
$$
## (b)
$$
\begin{align*}
    
    
    \\=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{\infin}r(s_t,a_t)
    \\=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{\infin}r(s_t,a_t)
\end{align*}
$$