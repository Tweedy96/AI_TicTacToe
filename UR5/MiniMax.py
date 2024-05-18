class MinimaxPlayer:
    def __init__(self):
        self.player = -1  # Player 1 (usually X)
        self.opponent = 1  # Player -1 (usually O)

    def is_moves_left(self, board):
        for row in board:
            if 0 in row:
                return True
        return False

    def evaluate(self, board):
        # Check rows for victory
        for row in range(3):
            if board[row][0] == board[row][1] == board[row][2]:
                if board[row][0] == self.player:
                    return 10
                elif board[row][0] == self.opponent:
                    return -10

        # Check columns for victory
        for col in range(3):
            if board[0][col] == board[1][col] == board[2][col]:
                if board[0][col] == self.player:
                    return 10
                elif board[0][col] == self.opponent:
                    return -10

        # Check diagonals for victory
        if board[0][0] == board[1][1] == board[2][2]:
            if board[0][0] == self.player:
                return 10
            elif board[0][0] == self.opponent:
                return -10

        if board[0][2] == board[1][1] == board[2][0]:
            if board[0][2] == self.player:
                return 10
            elif board[0][2] == self.opponent:
                return -10

        return 0

    def minimax(self, board, depth, is_max):
        score = self.evaluate(board)

        # If player has won the game return evaluated score
        if score == 10:
            return score - depth

        # If opponent has won the game return evaluated score
        if score == -10:
            return score + depth

        # If there are no more moves and no winner, it is a tie
        if not self.is_moves_left(board):
            return 0

        # If this is the maximizer's move
        if is_max:
            best = -1000

            # Traverse all cells
            for i in range(3):
                for j in range(3):
                    # Check if cell is empty
                    if board[i][j] == 0:
                        # Make the move
                        board[i][j] = self.player

                        # Call minimax recursively and choose the maximum value
                        best = max(best, self.minimax(board, depth + 1, not is_max))

                        # Undo the move
                        board[i][j] = 0
            return best

        # If this is the minimizer's move
        else:
            best = 1000

            # Traverse all cells
            for i in range(3):
                for j in range(3):
                    # Check if cell is empty
                    if board[i][j] == 0:
                        # Make the move
                        board[i][j] = self.opponent

                        # Call minimax recursively and choose the minimum value
                        best = min(best, self.minimax(board, depth + 1, not is_max))

                        # Undo the move
                        board[i][j] = 0
            return best

    def find_best_move(self, board, turn):
        best_val = -1000 if turn == self.player else 1000
        best_move = (-1, -1)

        # Traverse all cells, evaluate minimax function for all empty cells
        for i in range(3):
            for j in range(3):
                # Check if cell is empty
                if board[i][j] == 0:
                    # Make the move
                    board[i][j] = turn

                    # Compute evaluation function for this move
                    move_val = self.minimax(board, 0, turn == self.opponent)

                    # Undo the move
                    board[i][j] = 0

                    # If the value of the current move is more than the best value, update best
                    if (turn == self.player and move_val > best_val) or (turn == self.opponent and move_val < best_val):
                        best_move = (i, j)
                        best_val = move_val
                        
        best_move = best_move[0] * 3 + best_move[1]
        return best_move
