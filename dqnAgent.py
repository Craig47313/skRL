#using a slightly modified version of the code at https://github.com/krazyness/CRBot-public
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        self.action_size = action_size

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def act(self, state, possibleStates):
        if random.random() < self.epsilon: #do smth random
            valid_actions = torch.nonzero(torch.tensor(possibleStates)).flatten()
            return valid_actions[torch.randint(len(valid_actions), (1,))].item()

        state_t = torch.FloatTensor(state).unsqueeze(0)     
        with torch.no_grad():
            q_values = self.model(state_t).squeeze(0)   

        # Mask invalid actions all at once (vectorized)
        mask = torch.tensor(possibleStates, dtype=torch.bool)
        q_values[~mask] = float('-inf')

        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                next_q = self.target_model(torch.FloatTensor(next_state).unsqueeze(0)).max().item()
                target += self.gamma * next_q
            
            state_t = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state_t).squeeze(0).clone()
            target_f[action] = float(target)

            prediction = self.model(state_t).squeeze(0)[action]
            loss = self.criterion(prediction, target_f[action].detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, filename):
        # Look in models/ directory by default
        path = filename
        if not os.path.isabs(filename):
            path = os.path.join("models", filename)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Loaded model weights from {path}")