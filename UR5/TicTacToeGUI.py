import tkinter as tk
from tkinter import messagebox
from UR5Sim import UR5Sim
from TicTacToeAI import TicTacToeAI
import time

class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        master.title("Tic-Tac-Toe Robot Control")

        self.robot_simulator = UR5Sim()
        self.ai = TicTacToeAI()
        self.current_player = 'X'
        self.game_active = True

        self.board = [['' for _ in range(3)] for _ in range(3)]
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                btn = tk.Button(master, text='', font=('Arial', 20), height=2, width=4,
                                command=lambda i=i, j=j: self.handle_button_click(i, j))
                btn.grid(row=i, column=j, sticky="nsew", padx=2, pady=2)
                self.buttons[i][j] = btn

    def handle_button_click(self, row, col):
        if self.game_active and self.buttons[row][col]["text"] == "":
            self.make_move(row, col, self.current_player)

    def make_move(self, row, col, player):
        self.buttons[row][col].config(text=player)
        self.board[row][col] = player
        self.robot_simulator.move_robot(row, col, callback=lambda: self.after_robot_move(player))

    def after_robot_move(self, player):
        if self.check_winner():
            messagebox.showinfo("Game Over", f"{player} wins!")
            self.reset_game()
        elif self.is_board_full():
            messagebox.showinfo("Game Over", "It's a draw!")
            self.reset_game()
        else:
            self.switch_player()

    def switch_player(self):
        self.current_player = 'X' if self.current_player == 'O' else 'O'
        if self.current_player == 'O' and self.game_active:
            self.ai_move()

    def ai_move(self):
        if not self.game_active:
            return

        move = self.ai.find_random_move(self.board)
        if move:
            time.sleep(1)
            self.make_move(*move, 'O')

    def check_winner(self):
        # Check rows, columns, and diagonals for a winner
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != '':
                return True
            if self.board[0][i] == self.board[1][i] == self.board[2][i] != '':
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != '' or \
           self.board[0][2] == self.board[1][1] == self.board[2][0] != '':
            return True
        return False

    def is_board_full(self):
        # Check if the board is full (draw condition)
        for row in self.board:
            if '' in row:
                return False
        return True

    def reset_game(self):
        # Reset the game board
        for i in range(3):
            for j in range(3):
                self.buttons[i][j].config(text='')
                self.board[i][j] = ''
        self.current_player = 'X'  # X always starts
    
    def start(self):
        self.master.mainloop()
            
