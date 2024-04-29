from UR5Sim import UR5Sim
import AI
from TicTacToeGUI import TicTacToeGUI
from tkinter import Tk

def main():
    root = Tk()
    gui = TicTacToeGUI(root)
    gui.start()

if __name__ == "__main__":
    main()