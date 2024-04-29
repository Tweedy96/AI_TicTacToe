import tkinter as tk
from UR5Sim import UR5Sim

class TicTacToeGUI:
    def __init__(self, master):
        print("Starting Tic-Tac-Toe GUI")
        self.master = master
        master.title("Tic-Tac-Toe Robot Control")

        # Create a robot simulator
        self.robot_simulator = UR5Sim()

        # Track the current player
        self.current_player = 'X'

        # Create a 3x3 grid of buttons
        self.buttons = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            for j in range(3):
                btn = tk.Button(master, text='', font=('Arial', 20), height=2, width=4,
                                command=lambda i=i, j=j: self.handle_button_click(i, j))
                btn.grid(row=i, column=j, sticky="nsew", padx=2, pady=2)
                self.buttons[i][j] = btn

        # Make the grid cells expandable
        for i in range(3):
            master.grid_rowconfigure(i, weight=1)
            master.grid_columnconfigure(i, weight=1)

    def handle_button_click(self, row, col):
        button = self.buttons[row][col]
        if button["text"] == "":
            button.config(text=self.current_player)
            self.robot_simulator.move_robot(row, col)
            self.current_player = 'O' if self.current_player == 'X' else 'X'
    
    def start(self):
        self.master.mainloop()
