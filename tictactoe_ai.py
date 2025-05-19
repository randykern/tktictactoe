import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 9)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TicTacToeAI:
    def __init__(self, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TicTacToeNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.memory = []
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.load_model()

    def board_to_state(self, board):
        """Convert board to neural network input state."""
        state = []
        for cell in board:
            if cell == " ":
                state.append(0)
            elif cell == "O":  # computer's moves
                state.append(1)
            else:  # human's moves
                state.append(-1)
        return torch.FloatTensor(state).to(self.device)

    def get_move(self, board, available_moves):
        """Get move using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            # Exploration: random move
            return random.choice(available_moves)
        
        # Exploitation: use neural network
        state = self.board_to_state(board)
        with torch.no_grad():
            q_values = self.model(state)
        
        # Mask invalid moves with large negative values
        for i in range(9):
            if i not in available_moves:
                q_values[i] = float('-inf')
        
        return q_values.argmax().item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        """Train the model using experiences from memory."""
        if len(self.memory) < 32:  # minimum batch size
            return

        # Sample random batch from memory
        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update model
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self):
        """Save model and training state."""
        save_dict = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(save_dict, 'tictactoe_model.pth')

    def load_model(self):
        """Load model and training state if available."""
        if os.path.exists('tictactoe_model.pth'):
            save_dict = torch.load('tictactoe_model.pth',
                                  map_location=self.device)
            self.model.load_state_dict(save_dict['model_state'])
            self.optimizer.load_state_dict(save_dict['optimizer_state'])
            self.epsilon = save_dict['epsilon']
