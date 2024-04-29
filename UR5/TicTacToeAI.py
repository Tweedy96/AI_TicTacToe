import random

class TicTacToeAI:
    def __init__(self):
        pass

    def find_random_move(self, board):
        # Create a list of empty spots
        empty_spots = [(i, j) for i in range(3) for j in range(3) if board[i][j] == '']
        # Randomly select an empty spot
        if empty_spots:
            return random.choice(empty_spots)
        return None
