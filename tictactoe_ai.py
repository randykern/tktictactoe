import numpy as np
import random
import os

class TicTacToeNet:
    """Simple feedforward neural network implemented with NumPy."""
    def __init__(self, learning_rate=0.001):
        self.lr = learning_rate
        # Xavier/He style initialization for better stability
        self.W1 = np.random.randn(9, 64) * np.sqrt(2 / 9)
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, 32) * np.sqrt(2 / 64)
        self.b2 = np.zeros(32)
        self.W3 = np.random.randn(32, 9) * np.sqrt(2 / 32)
        self.b3 = np.zeros(9)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def forward(self, x):
        """Perform a forward pass and return activations for backprop."""
        z1 = x @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self.relu(z2)
        z3 = a2 @ self.W3 + self.b3
        return z3, (x, z1, a1, z2, a2)

    def predict(self, x):
        z3, _ = self.forward(x)
        return z3

    def train_batch(self, x, target):
        """Update network weights using a batch of target Q-values."""
        out, cache = self.forward(x)
        grad = 2 * (out - target) / x.shape[0]
        self.backward(grad, cache)

    def backward(self, grad_z3, cache):
        x, z1, a1, z2, a2 = cache
        grad_W3 = a2.T @ grad_z3
        grad_b3 = grad_z3.sum(axis=0)

        grad_a2 = grad_z3 @ self.W3.T
        grad_z2 = grad_a2 * (z2 > 0)
        grad_W2 = a1.T @ grad_z2
        grad_b2 = grad_z2.sum(axis=0)

        grad_a1 = grad_z2 @ self.W2.T
        grad_z1 = grad_a1 * (z1 > 0)
        grad_W1 = x.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)

        self.W3 -= self.lr * grad_W3
        self.b3 -= self.lr * grad_b3
        self.W2 -= self.lr * grad_W2
        self.b2 -= self.lr * grad_b2
        self.W1 -= self.lr * grad_W1
        self.b1 -= self.lr * grad_b1

    def state_dict(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3,
        }

    def load_state_dict(self, state):
        self.W1 = state['W1']
        self.b1 = state['b1']
        self.W2 = state['W2']
        self.b2 = state['b2']
        self.W3 = state['W3']
        self.b3 = state['b3']


class TicTacToeAI:
    def __init__(self, learning_rate=0.001):
        self.model = TicTacToeNet(learning_rate)
        self.memory = []
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.load_model()

    @staticmethod
    def board_to_state(board):
        """Convert board to neural network input state."""
        mapping = {' ': 0, 'O': 1, 'X': -1}
        return np.array([mapping[cell] for cell in board], dtype=float)

    def get_move(self, board, available_moves):
        """Get move using epsilon-greedy strategy."""
        if random.random() < self.epsilon:
            return random.choice(available_moves)

        state = self.board_to_state(board).reshape(1, -1)
        q_values = self.model.predict(state)[0]
        # Mask invalid moves with large negative values
        for i in range(9):
            if i not in available_moves:
                q_values[i] = -np.inf
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < 32:
            return

        batch = random.sample(self.memory, 32)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.stack(states)
        next_states = np.stack(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        current_q = self.model.predict(states)
        next_q = self.model.predict(next_states)

        target_q = current_q.copy()
        max_next = np.max(next_q, axis=1)
        target_values = rewards + (1 - dones) * self.gamma * max_next
        target_q[np.arange(32), actions] = target_values

        self.model.train_batch(states, target_q)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self):
        data = self.model.state_dict()
        data['epsilon'] = self.epsilon
        np.savez('tictactoe_model.npz', **data)

    def load_model(self):
        if os.path.exists('tictactoe_model.npz'):
            data = np.load('tictactoe_model.npz')
            state = {k: data[k] for k in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']}
            self.model.load_state_dict(state)
            self.epsilon = float(data['epsilon'])

