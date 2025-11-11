# Reinforcement Learning

## Formulation

### Basic Concept (`BaseAgent.py`, `RandomAgent.py`)

+ Environment: 負責產生 Reward

+ Agent: 負責吐出 Action ，根據 State, Reward 來決定如何修正吐出的 Action

+ State: 表示 Environment 用

```
       -> Environment ->
Action |               |  state, reward
       <-    Agent   <-
```

+ 通用訓練流程
```.py
def train():
    for e in |ep|:
        # 訓練幾 "episodes" 幾局遊戲
        total_reward
        state = env.reset()
        for t in |T|:
            # 一局遊戲內的一步 也可以玩到遊戲結束
            action = agent.select()
            next_state, reward = env.step(action)
            agent.update(state, action, reward)
            state = next_state
            total_reward += reward
    return total_reward
```

+ 原理

狀態s下的Except return (Value Function) => 知道什麼情況下Except return最高，知道怎麼做事
$$V(s)=\mathbb{E}[G_t|S_t=s]$$
就是return (其中隱藏隨機的過程:不同Action)
$$G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+...$$

不過實際上想要得到 Value Function 很困難， 因為不會知道Environment轉移機率
所以用Q Value替代 (多加上指定Action)
$$Q(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a]$$

+ 類型

|Type|on-policy|off-policy|
|-|-|-|
|解釋|update全部策略(greedy探索也包含)|只 update 最佳策略|
|差異|執行和update會相同|執行和update會不同|
|範例|MC, SARSA_TD| Q_learn_TD|
|特色|可能很慢 但穩定|訓練較快 不過可能震盪|

### Monte Carlo (`MCAgent.py`)
在玩完一整局之後才估計$Q(s,a)$
估計方法
$$Q(s,a)\approx \frac{1}{N}\sum_{i=1}^N G_i(s,a)$$

First-Visit MC : 同一局遊戲中 只有第一次遇到 $(s,a)$ 才更新 => 為了穩定

選action方法: $\epsilon$-greedy + $Q(s,a)$ 最大的 action

### State, Reward, Action, State, Reward with Tempeoral Different (`SARSA_TD.py`)

+ Termperoal Different
$TD(0)$ 表示 booststrap -> 只用下一步來來更新 Q Value
$TD(1)$ 就是 MC -> 用全部的結果跟新
$$V(s_t)=\mathbb{E}[R_{t+1}+\gamma (R_{t+2} + \gamma R_{t+1} + ...)]$$
Bellman equation
$$V(s_t)=\mathbb{E}[R_{t+1}+\gamma V(s_{t+1})]$$
所以定義Value Function為目前Value 加上誤差，誤差來源 Bellman Qquation
$$V(S_t)\leftarrow V(s_t)+\alpha [R_{t+1}+\gamma V(S_{t+1})-V(S_t)]$$

+ discretize state
把環境離散化 不然連續數值Q Table會太大

+ On-Policy
On-Policy在 就直接拿下一個$Q(S_{t+1}, A_{t+1})$來更新
$$Q(S_t,A_t)\leftarrow Q(S_t, S_a)+\alpha [R_{t+1}+\gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

### Q-Learning with Temperal Different (`QLearn_TD.py`)
+ Off-Policy 
update的時候只拿表現最好的那個來
$$Q(S_t,A_t)\leftarrow Q(S_t, S_a)+\alpha [R_{t+1}+\gamma \max_{a}Q(S_{t+1}, a) - Q(S_t, A_t)]$$

### Deep Q-Learning Network (`DQN.py`)

## Implementation

### Create Env
```bash
python -m venv DRL_ENV
source DRL_ENV/bin/activate
pip install -r requirement.txt
```

