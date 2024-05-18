import numpy as np
import gym
import time

class TicTacToeEnv:
    def __init__(self, simulate=False):
        self.board = np.zeros((3, 3), dtype=int)  # 3x3 grid
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int32)  # Flat array of 9 spaces
        self.action_space = gym.spaces.Discrete(9)  # 9 possible moves
        self.current_player = 1  # 1 for 'X', -1 for 'O'
        self.done = False
        self.simulate = simulate

    def reset(self):
        self.board.fill(0)
        self.done = False
        return self.board.flatten()
    
    def valid_actions(self):
        return [i for i in range(9) if self.board.flatten()[i] == 0]

    def step(self, action, player):
        self.current_player = player
        row, col = divmod(action, 3)
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            if self.check_winner():
                self.done = True
                if self.current_player == -1:
                    print("AI wins!")
                    reward = 50
                else:
                    print("AI loses!")
                    reward = -50
            elif self.is_board_full():
                self.done = True
                reward = 25  # Draw, neutral reward
                print("Its a draw!")
            else:
                reward = -5  # Encourage more moves
        else:
            reward = -50  # Penalize illegal moves less harshly
            print("Illegal move!")
        return self.board.flatten(), reward, self.done, {}

    def check_winner(self):
        # Check rows, columns, and diagonals for a win
        board = self.board.reshape(3, 3)  # Assuming self.board is a flat array
        for i in range(3):
            if abs(sum(board[i, :])) == 3:  # Check each row
                return True
            if abs(sum(board[:, i])) == 3:  # Check each column
                return True
        if abs(sum(board.diagonal())) == 3 or abs(sum(np.fliplr(board).diagonal())) == 3:  # Check diagonals
            return True
        return False
    
    def is_board_full(self):
        if np.any(self.board == 0):
            return False
        return True
    
    def get_state(self):
        return self.board.flatten()
    
    def close(self):
        pass