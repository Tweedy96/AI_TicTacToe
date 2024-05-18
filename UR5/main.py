from TicTacToeGUI import TicTacToeGUI
from tkinter import Tk
from DQN import DQN_Solver
from TicTacToeEnv import TicTacToeEnv
import torch
import matplotlib.pyplot as plt
import random
from MiniMax import MinimaxPlayer


REPLAY_START_SIZE = 500
EPISODES = 1000

best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
model_path = "tictactoe_minimax.pkl"

def train_dqn(episodes, agent, env):
    episode_batch_score = 0
    episode_reward = 0
    loss_history = []

    try:
        print("Model found, loading...")
        agent.policy_network.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print("No model found, training started...")

    player = 1 # human = 1 / ai = -1
    ai = MinimaxPlayer()

    for i in range(episodes):
        print("Episode: ", i)

        state = env.reset()
        done = False
        while not done:
            valid_actions = env.valid_actions()
            if player == 1:
                action = random.choice(valid_actions)
                # action = agent.choose_action(state, valid_actions)
            else:
                action = ai.find_best_move(env.board, -1)
            state_, reward, done, _ = env.step(action, player)
            player *= -1

            agent.memory.add(state, action, reward, state_, done)
            state = state_

            if agent.memory.mem_count > REPLAY_START_SIZE:
                agent.learn()

            episode_batch_score += reward
            episode_reward += reward  

        episode_history.append(i)
        episode_reward_history.append(episode_reward)
        episode_reward = 0

        if i % 100 == 0 and agent.memory.mem_count > REPLAY_START_SIZE:
            # torch.save(agent.policy_network.state_dict(), model_path)
            torch.save(agent.policy_network.state_dict(), model_path)
            print("average total reward per episode batch since episode ", i, ": ", episode_batch_score/ float(100))
            episode_batch_score = 0
            loss = agent.learn()
            loss_history.append(loss)
        elif agent.memory.mem_count < REPLAY_START_SIZE:
            episode_batch_score = 0

    # Calculate running average of reward
    running_avg_reward = []
    window_size = 50
    for i in range(len(episode_reward_history)):
        if i < window_size:
            running_avg_reward.append(sum(episode_reward_history[:i+1]) / (i+1))
        else:
            running_avg_reward.append(sum(episode_reward_history[i-window_size+1:i+1]) / window_size)

    plt.plot(episode_history, episode_reward_history)
    plt.plot(episode_history, running_avg_reward, color='red')
    plt.title('Reward vs. Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Loss vs. Training Steps')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
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