#!/usr/bin/env python3
import random
import argparse
import sys
from tictactoe_ai import TicTacToeAI

class TicTacToe:
    def __init__(self, ai):
        self.board = [" " for _ in range(9)]
        self.current_player = "X"
        self.winner = None
        self.game_over = False
        self.human_player = "X"
        self.computer_player = "O"
        self.ai = ai
        self.game_history = []
        self.simple_ai_mode = False

    def print_board(self):
        """Display the current state of the board."""
        print("\n")
        for i in range(0, 9, 3):
            print(f" {self.board[i]} | {self.board[i+1]} | {self.board[i+2]} ")
            if i < 6:
                print("-----------")
        print("\n")

    def print_position_guide(self):
        """Display the position guide for players."""
        print("\nPosition Guide:")
        for i in range(0, 9, 3):
            print(f" {i+1} | {i+2} | {i+3} ")
            if i < 6:
                print("-----------")
        print("\n")

    def make_move(self, position):
        """Make a move on the board."""
        if self.board[position] == " ":
            self.board[position] = self.current_player
            return True
        return False

    def get_available_moves(self):
        """Get list of available positions."""
        return [i for i, spot in enumerate(self.board) if spot == " "]

    def check_winner(self):
        """Check if there's a winner."""
        # Check rows
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] != " ":
                return self.board[i]

        # Check columns
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] != " ":
                return self.board[i]

        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8] != " ":
            return self.board[0]
        if self.board[2] == self.board[4] == self.board[6] != " ":
            return self.board[2]

        return None

    def is_board_full(self):
        """Check if the board is full."""
        return " " not in self.board

    def switch_player(self):
        """Switch the current player."""
        self.current_player = "O" if self.current_player == "X" else "X"

    def get_reward(self, winner):
        """Calculate reward for the AI."""
        if winner == self.computer_player:
            return 1.0
        elif winner == self.human_player:
            return -1.0
        else:
            return 0.5  # Draw is better than losing

    def get_winning_move(self, player):
        """Check if there's a winning move for the given player."""
        available_moves = self.get_available_moves()
        for move in available_moves:
            # Try the move
            self.board[move] = player
            winner = self.check_winner()
            # Undo the move
            self.board[move] = " "
            if winner == player:
                return move
        return None

    def get_simple_ai_move(self):
        """Get move for simple AI player."""
        # Try to win
        winning_move = self.get_winning_move(self.human_player)
        if winning_move is not None:
            return winning_move

        # Block opponent from winning
        blocking_move = self.get_winning_move(self.computer_player)
        if blocking_move is not None:
            return blocking_move

        # Take center if available
        if self.board[4] == " ":
            return 4

        # Take corners if available
        corners = [0, 2, 6, 8]
        available_corners = [corner for corner in corners if self.board[corner] == " "]
        if available_corners:
            return random.choice(available_corners)

        # Take any available edge
        edges = [1, 3, 5, 7]
        available_edges = [edge for edge in edges if self.board[edge] == " "]
        if available_edges:
            return random.choice(available_edges)

        # Take any available move
        return random.choice(self.get_available_moves())

    def play(self):
        """Main game loop."""
        print("Welcome to Tic Tac Toe!")
        if self.simple_ai_mode:
            print("Simple AI (X) vs Neural Network AI (O)")
        else:
            print("You are X, Neural Network AI is O")
            print("You go first!")
        print(f"Neural Network AI Exploration Rate: {self.ai.epsilon:.2f}")

        result = 0

        while not self.game_over:
            self.print_board()
            
            if self.current_player == self.human_player:
                if self.simple_ai_mode:
                    print("Simple AI's turn (X)")
                    position = self.get_simple_ai_move()
                    self.make_move(position)
                    print(f"Simple AI chose position {position + 1}")
                else:
                    self.print_position_guide()
                    print("Your turn (X)")
                    
                    while True:
                        try:
                            position = int(input("Enter position (1-9): ")) - 1
                            if 0 <= position <= 8:
                                if self.make_move(position):
                                    break
                                else:
                                    print("That position is already taken!")
                            else:
                                print("Please enter a number between 1 and 9!")
                        except ValueError:
                            print("Please enter a valid number!")
            else:
                print("Neural Network AI's turn (O)")
                state = self.ai.board_to_state(self.board)
                available_moves = self.get_available_moves()
                position = self.ai.get_move(self.board, available_moves)
                self.make_move(position)
                print(f"Neural Network AI chose position {position + 1}")

            # Store the state and action for training
            if self.current_player == self.computer_player:
                self.game_history.append((state, position))

            # Check for winner
            winner = self.check_winner()
            if winner:
                self.print_board()
                reward = self.get_reward(winner)
                
                # Update AI with game results
                for state, action in self.game_history:
                    self.ai.remember(state, action, reward, 
                                   self.ai.board_to_state(self.board), True)
                
                if winner == self.human_player:
                    if self.simple_ai_mode:
                        print("Simple AI wins!")
                    else:
                        print("You win!")
                    result = 1  # X wins
                else:
                    print("Neural Network AI wins!")
                    result = 2  # O wins
                self.game_over = True
                break

            # Check for draw
            if self.is_board_full():
                self.print_board()
                reward = self.get_reward(None)
                
                # Update AI with game results
                for state, action in self.game_history:
                    self.ai.remember(state, action, reward,
                                   self.ai.board_to_state(self.board), True)
                
                print("It's a draw!")
                result = 0  # Draw
                self.game_over = True
                break

            self.switch_player()

        # Train the AI after each game
        self.ai.train()
        self.ai.save_model()

        return result

def main():
    parser = argparse.ArgumentParser(description='Play Tic Tac Toe against AI')
    parser.add_argument('--mode', choices=['human', 'auto'], default='human',
                      help='Game mode: "human" to play yourself, "auto" to watch algorithmic vs AI')
    parser.add_argument('--iterations', type=int, default=1,
                      help='Number of games to play in auto mode')
    args = parser.parse_args()

    if args.mode == 'human' and args.iterations != 1:
        print("Warning: iterations parameter is ignored in human mode", file=sys.stderr)
        args.iterations = 1

    ai = TicTacToeAI()

    results = {0: 0, 1: 0, 2: 0}  # Draw, X wins, O wins
    for i in range(args.iterations):
        if args.iterations > 1:
            print(f"\nGame {i+1}/{args.iterations}")
        game = TicTacToe(ai)
        game.simple_ai_mode = (args.mode == 'auto')
        result = game.play()
        results[result] += 1

    if args.iterations > 1:
        print(f"\nResults after {args.iterations} games:", file=sys.stderr)
        print(f"Draws: {results[0]}", file=sys.stderr)
        print(f"X wins: {results[1]}", file=sys.stderr)
        print(f"O wins: {results[2]}", file=sys.stderr)

    # Return the result of the last game
    sys.exit(result)

if __name__ == "__main__":
    main()
