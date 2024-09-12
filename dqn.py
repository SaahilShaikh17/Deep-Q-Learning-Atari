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

    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        q_values = self(obs_t.unsqueeze(0))  # add batch dimension

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action


# Setting up the environment
env = gym.make('CartPole-v1')
replay_buffer = deque(maxlen=BUFFER_SIZE)
rew_buffer = deque([0, 0], maxlen=100)  # Reward buffer to store the rewards earned by the agent in a single episode.
episode_reward = 0.0

online_net = Network(env)
target_net = Network(env)
target_net.load_state_dict(online_net.state_dict())

optimizer = torch.optim.Adam(online_net.parameters(), lr=5e-4)

# Initialize Replay Buffer
obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()
    new_obs, rew, done, truncated, _ = env.step(action)
    done = done or truncated  # Combine termination signals
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    if done:
        obs = env.reset()

# Main Training Loop
obs = env.reset()
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])  # Epsilon decay

    if random.random() <= epsilon:
        action = env.action_space.sample()
    else:
        action = online_net.act(obs)

    new_obs, rew, done, truncated, _ = env.step(action)
    done = done or truncated  # Combine termination signals
    transition = (obs, action, rew, done, new_obs)
    replay_buffer.append(transition)
    obs = new_obs

    episode_reward += rew
    if done:
        obs = env.reset()
        rew_buffer.append(episode_reward)
        episode_reward = 0.0

    # Start Gradient Step
    transitions = random.sample(replay_buffer, BATCH_SIZE)

    obses = [t[0] for t in transitions]

    # If the observation is a single value or small vector, this will work
    obses = np.asarray(obses, dtype=np.float32)

    actions = np.asarray([t[1] for t in transitions])
    rews = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    new_obses = np.asarray([t[4] for t in transitions])

    # Convert to tensor
    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1)
    rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1)
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32)

    # Compute Targets
    target_q_values = target_net(new_obses_t)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

    # Compute Loss
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

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
        print()
        print('Step', step)
        print('Avg Reward', np.mean(rew_buffer))
