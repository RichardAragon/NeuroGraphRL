import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple
import cv2
from ale_py import ALEInterface, roms
from tqdm import tqdm
import networkx as nx
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'energy'])

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        action = int(experience.action)
        experience = Experience(experience.state, action, experience.reward, experience.next_state, experience.done, experience.energy)
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        experiences = random.sample(self.buffer, batch_size)
        return Experience(*zip(*experiences))

    def __len__(self):
        return len(self.buffer)

def preprocess_state(state):
    gray = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return torch.tensor(resized, dtype=torch.float32).unsqueeze(0) / 255.0

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4):
        super(GraphAttentionLayer, self).__init__()
        self.attention = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.activation = nn.Tanh()

    def forward(self, x, edge_index):
        x = self.attention(x, edge_index)
        return self.activation(x)

class GraphProcessingStream(nn.Module):
    def __init__(self):
        super(GraphProcessingStream, self).__init__()
        self.layer1 = GraphAttentionLayer(1, 16, heads=4)
        self.layer2 = GraphAttentionLayer(64, 32, heads=4)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long).to(x.device))
        return x

class EnergyConservingNN(nn.Module):
    def __init__(self, input_shape, n_actions, graph_data):
        super(EnergyConservingNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.conv_out_size = self._get_conv_out(input_shape)
        self.graph_stream = GraphProcessingStream()
        self.energy_stream = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        graph_feature_size = 128
        energy_feature_size = 32
        combined_feature_size = self.conv_out_size + graph_feature_size + energy_feature_size
        
        self.value_stream = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.graph_data = graph_data

    def _get_conv_out(self, shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            conv_out = self.conv(dummy_input)
            return int(np.prod(conv_out.size()[1:]))

    def forward(self, x, energy):
        batch_size = x.size(0)
        visual_features = self.conv(x).view(batch_size, -1)
        graph_features = self.graph_stream(self.graph_data.x, self.graph_data.edge_index)
        graph_features = graph_features.expand(batch_size, -1)
        energy_features = self.energy_stream(energy.view(batch_size, -1))
        combined = torch.cat([visual_features, graph_features, energy_features], dim=1)
        value = self.value_stream(combined)
        advantages = self.advantage_stream(combined)
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

def build_qbert_graph():
    G = nx.Graph()
    levels = 4
    node_mapping = {}
    node_id = 0
    for level in range(levels):
        for position in range(level + 1):
            G.add_node(node_id)
            node_mapping[(level, position)] = node_id
            node_id += 1
    for level in range(levels - 1):
        for position in range(level + 1):
            current = node_mapping[(level, position)]
            G.add_edge(current, node_mapping[(level + 1, position)])
            G.add_edge(current, node_mapping[(level + 1, position + 1)])
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.ones((G.number_of_nodes(), 1))
    return Data(x=x, edge_index=edge_index)

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ale = ALEInterface()
    ale.setBool('display_screen', True)
    ale.loadROM(roms.get_rom_path("qbert"))
    actions = ale.getMinimalActionSet()

    graph_data = build_qbert_graph().to(device)
    state_shape = (4, 84, 84)
    policy_net = EnergyConservingNN(state_shape, len(actions), graph_data).to(device)
    target_net = EnergyConservingNN(state_shape, len(actions), graph_data).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(100000)
    epsilon = 1.0
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 32
    min_replay_size = 1000
    fall_penalty = -50

    for episode in tqdm(range(500)):
        ale.reset_game()
        state_stack = deque([preprocess_state(ale.getScreenRGB()) for _ in range(4)], maxlen=4)
        energy = 100.0
        total_reward = 0

        while not ale.game_over() and energy > 0:
            state = torch.cat(list(state_stack), dim=0).unsqueeze(0).to(device)
            energy_tensor = torch.tensor([[energy]], dtype=torch.float32).to(device)

            if random.random() < epsilon:
                action_index = random.randint(0, len(actions) - 1)
            else:
                with torch.no_grad():
                    qvals = policy_net(state, energy_tensor)
                    action_index = qvals.argmax().item()
                    action_index = min(action_index, len(actions) - 1)

            action = actions[action_index]
            reward = ale.act(action)
            next_state = preprocess_state(ale.getScreenRGB())
            state_stack.append(next_state)
            done = ale.game_over()
            energy -= 0.1

            if done:
                reward += fall_penalty

            replay_buffer.push(Experience(state.squeeze(0), action_index, reward, torch.cat(list(state_stack), dim=0).squeeze(0), done, energy))

            if len(replay_buffer) >= min_replay_size:
                batch = replay_buffer.sample(batch_size)
                states = torch.stack(batch.state).to(device)
                actions_batch = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(device)
                rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
                next_states = torch.stack(batch.next_state).to(device)
                dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)
                energies = torch.tensor(batch.energy, dtype=torch.float32).unsqueeze(1).to(device)

                with torch.no_grad():
                    next_qvals = target_net(next_states, energies).max(1)[0].unsqueeze(1)
                    target_qvals = rewards + gamma * next_qvals * (1 - dones)

                qvals = policy_net(states, energies).gather(1, actions_batch)
                loss = nn.MSELoss()(qvals, target_qvals)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(epsilon * epsilon_decay, 0.01)

if __name__ == "__main__":
    run_training()
