import random
import torch
from DQN import Network

class TicTacToeAI:
    def __init__(self, env, model_path):
        self.env = env
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = Network(self.env)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        return model

    def choose_best_move(self, state):
        valid_actions = self.env.valid_actions()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            q_values = self.model(state)
        valid_q_values = q_values[0, valid_actions]
        action = valid_actions[torch.argmax(valid_q_values).item()]
        return action  # Convert flat index to row, col

    def find_random_move(self, board):
        # Create a list of empty spots
        empty_spots = [i for i in range(9) if board[i] == 0]
        # Randomly select an empty spot
        if empty_spots:
            return random.choice(empty_spots)
        return None
    
