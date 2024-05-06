import random
import torch
from DQN import Network

class TicTacToeAI:
    def __init__(self, env, model_path):
        self.env = env
        input_dim = env.observation_space.shape[0]  # Assuming this gives the correct dimension
        output_dim = env.action_space.n
        self.model = self.load_model(model_path, input_dim, output_dim)

    def load_model(self, model_path, input_dim, output_dim):
        model = Network(input_dim, output_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        return model

    def choose_best_move(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            q_values = self.model(state)
        action = torch.argmax(q_values).item()
        return action  # Convert flat index to row, col

    def find_random_move(self, board):
        # Create a list of empty spots
        empty_spots = [(i, j) for i in range(3) for j in range(3) if board[i][j] == '']
        # Randomly select an empty spot
        if empty_spots:
            return random.choice(empty_spots)
        return None
    
