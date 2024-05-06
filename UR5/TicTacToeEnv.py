import numpy as np
from UR5Sim import UR5Sim
import DQN
import gym
import torch

class TicTacToeEnv:
    def __init__(self, render=False):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1  # 1 for 'X', -1 for 'O'
        self.done = False
        self.robot = UR5Sim()
        self.observation_space = self.board.flatten()
        self.action_space = gym.spaces.Discrete(9)
        self.simulate = render

    def reset(self):
        self.board.fill(0)
        self.current_player = 1
        self.done = False
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            if self.check_winner():
                self.done = True
                reward = 1.0  # Reward for winning
            elif np.all(self.board != 0):
                self.done = True
                reward = 0  # Draw
            else:
                self.current_player *= -1
                reward = 0
        else:
            reward = -1  # Penalize invalid move
            self.done = True  # End the game on an invalid move
        if self.simulate:
            self.render(action)  # Update the robot's position after each move
        
        return self.board.flatten(), reward, self.done, {}

    def check_winner(self):
        # Check rows, columns, and diagonals for a win
        print
        board = self.board.reshape(3, 3)  # Assuming self.board is a flat array
        for i in range(3):
            if abs(sum(board[i, :])) == 3:  # Check each row
                return True
            if abs(sum(board[:, i])) == 3:  # Check each column
                return True
        if abs(sum(board.diagonal())) == 3 or abs(sum(np.fliplr(board).diagonal())) == 3:  # Check diagonals
            return True
        print("No winner")
        return False
    
    def is_board_full(self):
        # Check if the board is full (draw condition)
        print("Checking if board is full")
        print(self.board)
        if np.any(self.board == 0):
            print("Board is not full")
            return False
        print("Board is full")
        return True

       
    def render(self, position):
        self.robot.move_robot(position)

    def close(self):
        self.robot.close()
