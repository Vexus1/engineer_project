from dataclasses import dataclass
import sys

import pygame
from pygame import Surface
import chess
from icecream import ic

from gui.board import Board
from constants import *

@dataclass
class Interface:
    screen: Surface
    player: bool

    def __post_init__(self):
        self.start_new_game()

    def start_new_game(self) -> None:
        '''Starts a new game by resetting the board'''
        self.board = Board(self.screen)
        self.board.create_board() 
        self.board.initial_positions()
        self.chess_board = self.board.chess_board
        self.selected_square = None

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                ic(event)
                pos = pygame.mouse.get_pos()
                self.handle_click(pos)
        
    def handle_click(self, pos: tuple[int, int]) -> None:
        '''Handles a click on the board and processes moves'''
        col = pos[0] // TILE_SIZE
        row = 7 - (pos[1] // TILE_SIZE)
        square = chess.square(col, row)
        if self.selected_square is None:
            piece = self.chess_board.piece_at(square)
            if piece and piece.color == self.board.chess_board.turn:
                self.selected_square = square
        else:
            move = chess.Move(self.selected_square, square)
            if move in self.board.chess_board.legal_moves:
                self.board.chess_board.push(move)
                self.selected_square = None
                self.check_game_over()
            else:
                self.selected_square = None

    def check_game_over(self):
        '''Check if the game is over due to checkmate'''
        if self.chess_board.is_checkmate():
            self.start_new_game()

    def render(self):
        self.board.create_board()
        for i in range(64):
            piece = self.chess_board.piece_at(i)
            if piece:
                piece_symbol = piece.symbol()
                image = self.board.pieces_images[piece_symbol]
                selected_image = pygame.transform.scale(image, (TILE_SIZE,
                                                                TILE_SIZE))
                row = 7 - (i // 8)
                col = i % 8
                self.screen.blit(selected_image, (col * TILE_SIZE,
                                                  row * TILE_SIZE))
