# Cliff-walking-RL-agent

## SARSA Agent


SARSA (State-Action-Reward-State-Action) is an on-policy reinforcement learning algorithm. It learns the Q-values for state-action pairs by following a specific policy, meaning the Q-value updates are based on the actual actions the agent takes during training.

![image](https://github.com/user-attachments/assets/777559d6-636b-4d9a-bf84-db9e4b3f0bd5)

## Qlearning Agent


Q-Learning is an off-policy reinforcement learning algorithm. Unlike SARSA, it learns the Q-values based on the optimal action for each state, regardless of the policy currently being followed. This helps the agent find the best possible action for each state in the long run.

![image](https://github.com/user-attachments/assets/e4ed99ee-0a88-4560-a1dc-14da15a590e6)


## Random Agent


A Random Agent selects actions randomly from the available actions at each state, without any learning or optimization. It does not consider the environment’s feedback (reward) or previous experiences, making it suitable for testing baseline performance or benchmarking against learning agents.


![image](https://github.com/user-attachments/assets/b0fad943-3776-4dd9-9d1f-79ac3739f72c)
