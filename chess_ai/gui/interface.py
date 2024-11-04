from dataclasses import dataclass

import pygame
from pygame import Surface

from gui.board import Board

@dataclass
class Interface:
    screen: Surface
    board: Board = None

    def __post_init__(self):
        self.board = Board(self.screen)

    def render(self):
        self.board.create_board() 
        self.board.initial_positions()
