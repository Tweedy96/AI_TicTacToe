from TicTacToeGUI import TicTacToeGUI
from tkinter import Tk
from DQN import DQN_Solver
from TicTacToeEnv import TicTacToeEnv
import torch
import matplotlib.pyplot as plt

REPLAY_START_SIZE = 1000
EPISODES = 2000

best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
model_path = "tictactoe_dqn.pth"

def train_dqn(episodes, agent, env):
    episode_batch_score = 0
    episode_reward = 0
    total_reward = 0

    try:
        print("Model found, loading...")
        agent.policy_network.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        print("No model found, training started...")

    for i in range(episodes):
        print("Episode: ", i)
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
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

        episode_history.append(i)
        episode_reward_history.append(episode_reward)
        episode_reward = 0

        if i % 100 == 0 and agent.memory.mem_count > REPLAY_START_SIZE:
            torch.save(agent.policy_network.state_dict(), model_path)
            print("average total reward per episode batch since episode ", i, ": ", episode_batch_score/ float(100))
            episode_batch_score = 0
        elif agent.memory.mem_count < REPLAY_START_SIZE:
            episode_batch_score = 0
        
    plt.plot(episode_history, episode_reward_history)
    plt.title('Reward vs. Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    env.close()
  
def main():
    training = False
    simulate = True

    env = TicTacToeEnv(simulate)  # Initialize the Tic-Tac-Toe environment
    agent = DQN_Solver(env)

    if training:
        train_dqn(EPISODES, agent, env)
    else:
        root = Tk()
        gui = TicTacToeGUI(root, agent, env, model_path, simulate)
        gui.start()

if __name__ == '__main__':
    print("Starting training...")
    main()