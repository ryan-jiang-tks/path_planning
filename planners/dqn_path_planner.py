import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

class DQNPathPlanner:
    def __init__(self, voxel_grid, start, goal, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 gamma=0.99, learning_rate=0.001, memory_size=10000,
                 batch_size=64):
        self.voxel_grid = voxel_grid
        self.start = np.array(start)
        self.goal = np.array(goal)
        
        # DQN parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.batch_size = batch_size
        
        # State and action space
        self.state_size = 9  # current_pos(3) + goal(3) + nearest_obstacle(3)
        self.action_size = 26  # 3D movements (including diagonals)
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net = DQN(self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
        # Define possible actions (26 directions in 3D)
        self.actions = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    if x == 0 and y == 0 and z == 0:
                        continue
                    self.actions.append(np.array([x, y, z]))

    def get_state(self, position):
        # Find nearest obstacle
        min_dist = float('inf')
        nearest_obstacle = np.zeros(3)
        
        # Sample points around current position
        for x in range(-2, 3):
            for y in range(-2, 3):
                for z in range(-2, 3):
                    check_pos = position + np.array([x, y, z])
                    if all(0 <= p < s for p, s in zip(check_pos, self.voxel_grid.shape)):
                        if self.voxel_grid[tuple(check_pos)]:
                            dist = np.linalg.norm(check_pos - position)
                            if dist < min_dist:
                                min_dist = dist
                                nearest_obstacle = check_pos
        
        # Normalize positions
        grid_size = np.array(self.voxel_grid.shape)
        pos_norm = position / grid_size
        goal_norm = self.goal / grid_size
        obs_norm = nearest_obstacle / grid_size
        
        return np.concatenate([pos_norm, goal_norm, obs_norm])

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append(Experience(state, action, reward, next_state, done))

    def train(self, num_episodes=1000):
        best_path = None
        min_path_length = float('inf')
        
        for episode in range(num_episodes):
            current_pos = np.array(self.start)
            path = [tuple(current_pos)]
            total_reward = 0
            
            while True:
                state = self.get_state(current_pos)
                action_idx = self.select_action(state)
                action = self.actions[action_idx]
                
                # Take action
                next_pos = current_pos + action
                
                # Check if valid move
                if not all(0 <= p < s for p, s in zip(next_pos, self.voxel_grid.shape)):
                    reward = -1
                    done = True
                elif self.voxel_grid[tuple(next_pos)]:
                    reward = -1
                    done = True
                elif np.array_equal(next_pos, self.goal):
                    reward = 100
                    done = True
                else:
                    reward = -0.1 - 0.01 * np.linalg.norm(next_pos - self.goal)
                    done = len(path) > 100  # Prevent too long paths
                
                next_state = self.get_state(next_pos)
                self.store_experience(state, action_idx, reward, next_state, done)
                
                if len(self.memory) > self.batch_size:
                    self._update_network()
                
                current_pos = next_pos
                path.append(tuple(current_pos))
                total_reward += reward
                
                if done:
                    break
            
            # Update epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update best path
            if total_reward > -50 and len(path) < min_path_length:
                min_path_length = len(path)
                best_path = path
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")
        
        return best_path

    def _update_network(self):
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Periodically update target network
        if random.random() < 0.01:
            self.target_net.load_state_dict(self.policy_net.state_dict())

def plan_path_dqn(voxel_grid, start, goal, num_episodes=1000):
    planner = DQNPathPlanner(voxel_grid, start, goal)
    path = planner.train(num_episodes=num_episodes)
    return path
