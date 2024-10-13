import gym
import numpy as np
import cv2

env = gym.make('CliffWalking-v0')

num_states = env.observation_space.n  
num_actions = env.action_space.n  
Q = np.zeros((num_states, num_actions)) 

num_episodes = 1000 
alpha = 0.1  
gamma = 0.99  
epsilon = 1.0  
epsilon_min = 0.1 
epsilon_decay = 0.995  

cliff_color = (0, 0, 0)  
goal_color = (0, 255, 0)  
agent_color = (0, 0, 255)  
start_color = (255, 0, 0)  
grid_color = (200, 200, 200)  

def visualize_cliff_walking(state):
    """Visualizes the CliffWalking environment with the agent's position."""
    img = np.ones((4, 12, 3), dtype=np.uint8) * 200

    img[3, 1:11] = cliff_color

    row = state // 12
    col = state % 12

    img[row, col] = agent_color

    img[3, 11] = goal_color 

    img_resized = cv2.resize(img, (600, 200), interpolation=cv2.INTER_NEAREST)
    for i in range(1, 4):
        cv2.line(img_resized, (0, i * 50), (600, i * 50), (0, 0, 0), 1)  
    for j in range(1, 12):
        cv2.line(img_resized, (j * 50, 0), (j * 50, 200), (0, 0, 0), 1) 

    cv2.imshow("Qlearnig_agent(off policy)", img_resized)
    cv2.waitKey(100) 

def choose_action(state, epsilon):
    """Selects an action using the epsilon-greedy strategy."""
    if np.random.rand() < epsilon:  
        return env.action_space.sample()
    else:  
        return np.argmax(Q[state])

for episode in range(num_episodes):
    
    state_info = env.reset()
    state = state_info[0] if isinstance(state_info, tuple) else state_info  
    state = int(state)  

    done = False  

    while not done:
        visualize_cliff_walking(state)  

        action = choose_action(state, epsilon)

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_state = int(next_state)  

        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

        done = terminated or truncated

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

cv2.destroyAllWindows()  
