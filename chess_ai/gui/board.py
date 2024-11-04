from dataclasses import dataclass
import os

import pygame
from pygame import Surface
import chess
from icecream import ic

from gui.pieces import Pieces
from constants import *

@dataclass
class Board:
    screen: Surface

    def __post_init__(self):
        self.board = chess.Board()
        self.pieces = Pieces(None, None)
        self.pieces_images = self.pieces.load_piece_images()
        self.pieces_images = self.pieces.scale_piece_images(self.pieces_images)

    def create_board(self) -> None:
        '''Creates an 8x8 grid of the chessboard.'''
        self.screen.fill((0, 0, 0))  # Czyszczenie ekranu
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BROWN
                pygame.draw.rect(self.screen, color, 
                                 (col * TILE_SIZE,
                                  row * TILE_SIZE,
                                  TILE_SIZE, TILE_SIZE))
        
    def initial_positions(self) -> None:
        '''Creates the initial position of pieces on the board.'''
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece:
                piece_symbol = piece.symbol()
                image = self.pieces_images[piece_symbol]
                row = 7 - (i // 8)  # Poprawiona linia
                col = i % 8
                self.screen.blit(image, (col * TILE_SIZE, row * TILE_SIZE))
