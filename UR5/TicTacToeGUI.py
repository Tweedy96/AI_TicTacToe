import tkinter as tk
from tkinter import messagebox
from TicTacToeEnv import TicTacToeEnv
from TicTacToeAI import TicTacToeAI  # Assuming this class is updated to use DQN

class TicTacToeGUI:
    def __init__(self, master, env, model_path):
        self.master = master
        master.title("Tic-Tac-Toe")
        self.env = env
        self.ai = TicTacToeAI(env, model_path)  # Initialize the AI with the path to the trained model
        self.current_player = 'X'  # Start with player X
        self.game_active = True

        # Setup GUI components (buttons, reset button)
        self.setup_gui()

    def setup_gui(self):
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                btn = tk.Button(self.master, text='', font=('Arial', 20), height=2, width=4,
                                command=lambda i=i, j=j: self.handle_button_click(i, j))
                btn.grid(row=i, column=j, sticky="nsew", padx=2, pady=2)
                self.buttons[i][j] = btn
        self.reset_button = tk.Button(self.master, text='Reset', command=self.reset_game)
        self.reset_button.grid(row=3, column=0, columnspan=3, sticky="nsew")

    def handle_button_click(self, i, j):
        if self.game_active and self.buttons[i][j]["text"] == "":
            self.make_move(i, j, self.current_player)
            if self.current_player == 'O':
                self.ai_move()  # Trigger AI move if it's 'O's turn

    def make_move(self, i, j, player):
        self.buttons[i][j].config(text=player)
        self.env.board[i][j] = 1 if player == 'X' else -1
        self.check_game_status()

    def ai_move(self):
        # AI finds the best move
        move = self.ai.choose_best_move(self.env.board.flatten())
        if move:
            self.make_move(*move, 'O')

    def check_game_status(self):
        if self.env.check_winner():
            messagebox.showinfo("Game Over", f"{self.current_player} wins!")
            self.game_active = False
        elif self.env.is_board_full():
            messagebox.showinfo("Game Over", "It's a draw!")
            self.game_active = False
        self.switch_player()

    def switch_player(self):
        self.current_player = 'O' if self.current_player == 'X' else 'X'

    def reset_game(self):
        self.env.reset()
        self.game_active = True
        for row in self.buttons:
            for btn in row:
                btn.config(text='')

    def start(self):
        self.master.mainloop()
