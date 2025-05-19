# Console Tic Tac Toe

A simple command-line implementation of the classic Tic Tac Toe game in Python.

## Features
- Interactive console-based gameplay
- Clear board visualization
- Input validation
- Win detection
- Player turn management

## How to Play
1. Install Dependencies:
    ```
    pip install -r requirements.txt
    ```

2. Run the game using Python 3:
   ```
   python tic_tac_toe.py
   ```
3. The game board is represented by a 3x3 grid with positions numbered 1-9
4. Players take turns entering numbers 1-9 to place their mark (X or O)
5. First player to get three in a row (horizontally, vertically, or diagonally) wins
6. If all spaces are filled with no winner, the game ends in a draw

## Requirements
- Python 3.x 

## Code history
I've completely revamped the game to use a neural network-based AI that learns from each game. Here are the key changes and features:

1. Added a neural network AI module (tictactoe_ai.py) that includes:
- A 3-layer neural network (9 → 64 → 32 → 9 neurons)
- Experience replay memory for learning
- Epsilon-greedy exploration strategy
- Model saving and loading capabilities
2. Updated the main game to:
- Use the neural network AI instead of the rule-based opponent
- Track game history for training
- Implement a reward system
- Train the AI after each game
- Save the AI's progress
3. The AI features:
- Reinforcement learning using Q-learning
- Exploration vs exploitation balance (starts with 100% exploration)
- Learning from both wins and losses
- Persistent learning across game sessions

The AI will:
- Start with random moves (high exploration rate)
- Learn from each game it plays
- Gradually become more strategic (exploration rate decreases)
- Save its progress between sessions
- Get better the more you play against it
The learning process:
- Wins give a reward of 1.0
- Losses give a reward of -1.0
- Draws give a reward of 0.5
- The AI uses these rewards to adjust its strategy
- The exploration rate (epsilon) decreases over time, making the AI more confident in its learned strategy
You'll notice:
- The AI starts making mostly random moves
- As you play more games, it will start making more strategic moves
- The exploration rate is displayed at the start of each game
- The AI's progress is saved between sessions