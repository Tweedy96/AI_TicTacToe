from TicTacToeGUI import TicTacToeGUI
from tkinter import Tk
from DQN import DQN_Solver
from TicTacToeEnv import TicTacToeEnv
import torch
import matplotlib.pyplot as plt
from TicTacToeAI import TicTacToeAI
from enum import Enum
import random


REPLAY_START_SIZE = 1000
EPISODES = 1000

best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
model_path = "tictactoe_dqn.pkl"


def train_dqn(episodes, agent, env):
    episode_batch_score = 0
    episode_reward = 0
    total_reward = 0
    wins = 0
    losses = 0
    draws = 0

    try:
        print("Model found, loading...")
        agent.policy_network.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print("No model found, training started...")

    for i in range(episodes):
        print("Episode: ", i)
        state = env.reset()
        player = 1 # human = 1 / ai = -1
        while True:
            if player == 1:
                valid_actions = env.valid_actions()
                action = random.choice(valid_actions)
            else:
                action = agent.choose_action(env, state)

            next_state, reward, done, _ = env.step(action, player)
            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if agent.memory.mem_count > REPLAY_START_SIZE:
                agent.learn()

            state = state
            episode_batch_score += reward
            episode_reward += reward

            if done:
                break

            player *= -1

        episode_history.append(i)
        episode_reward_history.append(episode_reward)
        episode_reward = 0
        if reward == 50:
            wins += 1
        elif reward == -50:
            losses += 1
        else:
            draws += 1

        if i % 100 == 0 and agent.memory.mem_count > REPLAY_START_SIZE:
            # torch.save(agent.policy_network.state_dict(), model_path)
            torch.save(agent.policy_network.state_dict(), model_path)
            print("average total reward per episode batch since episode ", i, ": ", episode_batch_score/ float(100))
            episode_batch_score = 0
        elif agent.memory.mem_count < REPLAY_START_SIZE:
            episode_batch_score = 0
    
    print("Wins: ", wins)
    print("Losses: ", losses)
    print("Draws: ", draws)

    plt.plot(episode_history, episode_reward_history)
    plt.title('Reward vs. Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    env.close()
  
def main():
    training = True
    simulate = ~training

    env = TicTacToeEnv(simulate)  # Initialize the Tic-Tac-Toe environment
    agent = DQN_Solver(env)

    if training:
        print("Starting training...")
        train_dqn(EPISODES, agent, env)
    else:
        root = Tk()
        gui = TicTacToeGUI(root, agent, env, model_path, simulate)
        gui.start()

if __name__ == '__main__':
    main()