from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random

# Hyperparameters
GAMMA = 0.99  # Discount rate to compute temporal difference
BATCH_SIZE = 32
BUFFER_SIZE = 50000  # Max number of transitions to be stored
MIN_REPLAY_SIZE = 1000  # How many transitions we want in the buffer before we start computing gradients
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000
LEARNING_RATE = 5e-4


class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n))

    def forward(self, x):
        return self.net(x)

    def act(self, obs, epsilon=0.0):
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            return env.action_space.sample()

        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))  # Add batch dimension
        max_q_index = torch.argmax(q_values, dim=1)[0]
        return max_q_index.detach().item()


# Setting up the environment
env = gym.make('CartPole-v1')
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque(maxlen=100)  # Store rewards for each episode
episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)
target_net.load_state_dict(online_net.state_dict())
optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

# Initialize Replay Buffer
obs, _ = env.reset()  # Newer gym versions return (obs, info)
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, done, truncated, _ = env.step(action)
    done = done or truncated  # Combine termination signals
    replay_buffer.append((obs, action, rew, done, new_obs))
    obs = new_obs if not done else env.reset()[0]

# Main Training Loop
obs, _ = env.reset()
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])  # Epsilon decay
    action = online_net.act(obs, epsilon)

    new_obs, rew, done, truncated, _ = env.step(action)
    done = done or truncated  # Combine termination signals
    replay_buffer.append((obs, action, rew, done, new_obs))
    obs = new_obs if not done else env.reset()[0]

    episode_reward += rew
    if done:
        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    # Start Gradient Step if replay buffer has enough transitions
    if len(replay_buffer) >= BATCH_SIZE:
        transitions = random.sample(replay_buffer, BATCH_SIZE)
        obses, actions, rews, dones, new_obses = zip(*transitions)

        # Convert to tensor
        obses_t = torch.as_tensor(obses, dtype=torch.float32)
        actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

        # Compute Targets
        with torch.no_grad():  # Detach to avoid backpropagation through target net
            target_q_values = target_net(new_obses_t)
            max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
            targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = online_net(obses_t)
        action_q_values = torch.gather(q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        # Gradient Descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # Logging
    if step % 1000 == 0:
        print(f'Step {step}, Avg Reward: {np.mean(rew_buffer):.2f}')
