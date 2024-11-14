# NeuroGraphRL

**NeuroGraphRL** is a novel reinforcement learning (RL) framework that integrates **graph neural networks (GNNs)**, **convolutional neural networks (CNNs)**, and **energy-conserving strategies** to train AI agents on complex environments, such as Atari games. This framework leverages the power of **graph-based processing** and **visual feature extraction** to create smarter and more efficient agents that understand both the spatial and structural elements of their environment.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
NeuroGraphRL combines three powerful techniques in one unified framework:
1. **Graph Neural Networks (GNNs)**: Uses Graph Attention Networks (GATs) to process spatial and relational information of environments, like Q*bert, by representing game levels as graphs.
2. **Convolutional Neural Networks (CNNs)**: Extracts visual features from game frames to help the agent make informed decisions.
3. **Energy-Conserving Strategies**: Introduces the concept of energy conservation to encourage agents to make more efficient moves and avoid self-destructive behaviors.

The framework is designed to train agents that can learn not just from pixel-based inputs but also from the structural layout of their environment, providing a deeper understanding of how actions affect outcomes.

## Features
- **Hybrid Neural Network Architecture**: Combines CNNs for visual processing with GNNs for understanding spatial structures.
- **Energy-Based Reward System**: Encourages agents to conserve energy and avoid harmful actions, such as jumping off platforms in games like Q*bert.
- **Replay Buffer for Experience Replay**: Stores past experiences to accelerate learning by sampling from a diverse set of states.
- **Epsilon-Greedy Policy**: Balances exploration and exploitation to optimize agent performance.
- **Graph Representation of Game Levels**: Leverages networkx and torch-geometric to represent game environments as graphs.

## Architecture
The architecture consists of:
- **Convolutional Neural Network (CNN)**: Extracts features from raw game frames.
- **Graph Attention Networks (GATs)**: Processes game levels represented as graphs.
- **Energy-Conserving Module**: Monitors the agent's energy level and penalizes wasteful actions.
- **Dueling Q-Network**: Splits the neural network into value and advantage streams to improve the stability of Q-learning.

### Model Diagram
```
+---------------------------+
|        Input Frames       |
+---------------------------+
            |
            v
+---------------------------+
|     Convolutional Layer   |
+---------------------------+
            |
            v
+---------------------------+
|  Graph Attention Network  |
+---------------------------+
            |
            v
+---------------------------+
|   Energy-Conserving MLP   |
+---------------------------+
            |
            v
+---------------------------+
|    Dueling Q-Network      |
+---------------------------+
            |
            v
+---------------------------+
|     Action Selection      |
+---------------------------+
```

## Installation
To install and run NeuroGraphRL, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/NeuroGraphRL.git
   cd NeuroGraphRL
   ```

2. **Install dependencies**:
   Make sure you have Python 3.10+ installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Atari ROMs**:
   You will need to download the necessary Atari ROMs. For Q*bert:
   ```bash
   pip install ale-py roms
   ```

4. **Run the training script**:
   ```bash
   python geometryqbert.py
   ```

## Usage
You can train an agent using the provided script:

```bash
python geometryqbert.py
```

Modify hyperparameters, such as the learning rate, epsilon decay, and reward functions, in the script to optimize the agent's performance.

## Project Structure
```
NeuroGraphRL/
├── geometryqbert.py         # Main training script
├── README.md                # Project documentation
├── requirements.txt         # List of dependencies
└── models/
    ├── energy_conserving_nn.py
    ├── graph_processing.py
    └── replay_buffer.py
```

## Results
The initial results demonstrate that the **energy-conserving reward system** and **graph-based spatial understanding** significantly improve the agent's performance compared to traditional DQN-based agents. Agents trained with NeuroGraphRL show better decision-making and longer survival times in complex environments.

### Sample Training Output
```
Episode 1: Total Reward: 549
Episode 2: Total Reward: 610
Episode 3: Total Reward: 720
...
```

## Contributing
We welcome contributions to NeuroGraphRL! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---


