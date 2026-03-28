# 🤖 Summer Project — Reinforcement Learning

> A hands-on exploration of core Reinforcement Learning algorithms, from tabular methods to deep policy-gradient approaches, implemented from scratch and benchmarked on standard environments.

---

## Overview

This summer project serves as a structured deep-dive into **Reinforcement Learning (RL)** — covering the theoretical foundations and practical implementations of key algorithms. The goal is to build intuition by coding agents that learn through interaction, observe their behaviour, and understand the trade-offs between different RL paradigms.

---

## Topics Covered

| Paradigm | Algorithms Implemented |
|---|---|
| **Tabular RL** | Q-Learning, SARSA, Monte Carlo Control |
| **Function Approximation** | Linear FA with Tile Coding |
| **Deep RL** | DQN (Deep Q-Network) |
| **Policy Gradient** | REINFORCE, Actor-Critic (A2C) |
| **Exploration** | ε-greedy, UCB, Thompson Sampling |

---

## Environments

Agents are trained and evaluated on:

- **OpenAI Gymnasium** classic control tasks — `CartPole-v1`, `MountainCar-v0`, `LunarLander-v2`  
- Custom grid-world environments for tabular method demonstrations  

---

## Project Structure

```
Summer-Project-1/
├── envs/
│   └── gridworld.py          # Custom grid environment
├── agents/
│   ├── q_learning.py
│   ├── sarsa.py
│   ├── monte_carlo.py
│   ├── dqn.py
│   └── policy_gradient.py
├── notebooks/
│   ├── 01_tabular_methods.ipynb
│   ├── 02_deep_q_network.ipynb
│   └── 03_policy_gradients.ipynb
├── utils/
│   ├── replay_buffer.py
│   └── plotting.py
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/anujpanwarma2024/Summer-Project-1.git
cd Summer-Project-1

# 2. Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a quick training demo
python agents/q_learning.py --env CartPole-v1 --episodes 500

# 5. Or explore the notebooks
jupyter notebook notebooks/
```

---

## Key Concepts Explored

### Bellman Equation (Q-Learning)
$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

### Policy Gradient (REINFORCE)
$$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ G_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

### DQN Enhancements
- Experience Replay (uniform sampling from replay buffer)  
- Target Network (soft updates for training stability)  

---

## Results

| Algorithm | Environment | Avg. Return (last 100 eps) |
|---|---|---|
| Q-Learning | GridWorld (5×5) | 0.92 |
| DQN | CartPole-v1 | 487 / 500 |
| REINFORCE | LunarLander-v2 | 215 |
| A2C | LunarLander-v2 | 248 |

---

## References

- Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.  
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* Nature.  
- Williams, R. J. (1992). *Simple statistical gradient-following algorithms for connectionist RL.*

---

## Author

**Anuj Panwar** · [GitHub](https://github.com/anujpanwarma2024)
