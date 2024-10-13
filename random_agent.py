import gym
import numpy as np
import cv2

env = gym.make('CliffWalking-v0')

def visualize_cliff_walking(state):
    """Visualizes the agent and environment."""
    img = np.ones((4, 12, 3), dtype=np.uint8) * 255

    cliff_color = (255, 0, 255)  
    goal_color = (0, 255, 0)     
    start_color = (255, 0, 0)    
    agent_color = (0, 0, 255)    

    img[3, 1:11] = cliff_color

    row = state // 12
    col = state % 12

    img[row, col] = agent_color

    img[3, 11] = goal_color  

    cv2.putText(img, 'S', (5, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Start
    cv2.putText(img, 'G', (555, 115), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Goal

    img_resized = cv2.resize(img, (600, 200), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("random_agent", img_resized)
    cv2.waitKey(100)

num_episodes = 5

for episode in range(num_episodes):
    state, _ = env.reset()  
    done = False

    print(f"Episode {episode + 1}:")

    while not done:
        visualize_cliff_walking(state)  

        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)  

        print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")

        state = next_state
        done = terminated or truncated  

cv2.destroyAllWindows() 
