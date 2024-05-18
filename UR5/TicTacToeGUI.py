import tkinter as tk
from tkinter import messagebox
from TicTacToeAI import TicTacToeAI  # Assuming this class is updated to use DQN
from DQN import DQN_Solver
from UR5Sim import UR5Sim
import pybullet
import time

REPLAY_START_SIZE = 1000

class TicTacToeGUI:
    def __init__(self, master, agent, env, model_path, simulate):
        self.master = master
        master.title("Tic-Tac-Toe")
        self.env = env
        self.ai = TicTacToeAI(env, model_path)  # Initialize the AI with the path to the trained model
        self.agent = agent
        self.current_player = 'X'  # Start with player X
        self.game_active = True
        self.robot = self.connect_to_robot()         
        self.simulate = simulate
        self.setup_gui()

    def connect_to_robot(self):
        if pybullet.isConnected():  # Check if there is an existing connection
            pybullet.disconnect()  # Disconnect if there is

        return UR5Sim()  # Establish a new connection and return the robot

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
        if self.simulate:
            position = 3 * i + j
            self.robot.move_robot(position)
            time.sleep(1)
        self.check_game_status()

    def ai_move(self):
        # Get current state from environment
        state = self.env.get_state()
        # Decide action
        action = self.ai.choose_best_move(state)
        # Apply action to the environment
        next_state, reward, done, _ = self.env.step(action, -1)
        # Save transition to replay buffer
        # self.agent.memory.add(state, action, reward, next_state, done)
        # Learn from the buffer if conditions are met
        # if self.agent.memory.mem_count > REPLAY_START_SIZE:
        #     self.agent.learn()
        self.make_move(*divmod(action, 3), 'O')
    

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
