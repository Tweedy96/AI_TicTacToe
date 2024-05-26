# UR5 Tic-Tac-Toe Simulation
This project implements a simulation of the UR5E robot playing Tic-Tac-Toe using reinforcement learning in PyBullet.

## Overview
This project simulates a UR5E robot arm that plays Tic-Tac-Toe. The robot uses a Deep Q-Network (DQN) to learn optimal strategies for the game. The goal is to train the robot to play Tic-Tac-Toe against a human or another AI, making strategic decisions to win or draw the game.

## Features
UR5E Robot Simulation: The UR5E robot arm is simulated in PyBullet, performing moves on a Tic-Tac-Toe board.
Reinforcement Learning: The robot is trained using a DQN to make optimal moves.
Training Visualization: Real-time visualization of training progress and reward metrics.
Error Handling: Ensures the robot only makes valid moves during the game.
How to Run
To run the simulation, execute the following command:

### Copy code
python3 UR5Sim.py
Dependencies
Python 3.8+
PyBullet
NumPy
PyTorch

### You can install the required dependencies using the following command:

pip install pybullet numpy torch

## Project Structure
UR5Sim.py: Main script to run the UR5E Tic-Tac-Toe simulation.
TicTacToeEnv.py: Environment setup for the Tic-Tac-Toe game.
DQN_Solver.py: Implementation of the Deep Q-Network and training logic.
TicTacToeGUI.py: GUI for visualizing the game and robot moves.

## DQN Hyperparameters
The following hyperparameters were used for training the DQN:

Learning Rate: 0.00025
Memory Size: 50000
Replay Start Size: 10000
Batch Size: 32
Gamma (Discount Factor): 0.99
Epsilon Start: 0.1
Epsilon End: 0.0001
Epsilon Decay: 4 * Memory Size
Network Update Iters: 5000
FC1 Dims: 64
FC2 Dims: 64
Demo
Embed demo video here

## CNN for Image Detection
Initially, we attempted to introduce a Convolutional Neural Network (CNN) for image detection to recognize the state of the Tic-Tac-Toe board. The idea was to allow the robot to interpret the game board visually and make moves based on this interpreted state. However, due to challenges in achieving reliable image detection, we decided to omit this feature from the final project.

## Future Work
Improve AI Strategy: Further tuning of hyperparameters and reward structure to enhance the AI's performance.
Reintroduce CNN: Continue development on the CNN for image detection to integrate visual game state recognition in the future.
Advanced Game Modes: Explore more complex game modes or multi-agent scenarios.

## Acknowledgements
We would like to thank the community and resources that have contributed to this project, making it possible to integrate advanced AI techniques with robotic simulation.
