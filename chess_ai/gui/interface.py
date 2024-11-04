from dataclasses import dataclass
import sys

import pygame
from pygame import Surface
import pygamepopup
from pygamepopup.menu_manager import MenuManager, InfoBox
from pygamepopup.components import Button
import chess
from chess import Move, Piece
from icecream import ic

from gui.board import Board
from constants import *

@dataclass
class Interface:
    screen: Surface
    player: bool

    def __post_init__(self):
        self.start_new_game()
        pygamepopup.init()
        self.menu_manager = MenuManager(screen=self.screen)

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
                pos = pygame.mouse.get_pos()
                if self.menu_manager.active_menu:
                    self.menu_manager.click(event.button, event.pos)
                else:
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
            if self.is_promotion(move):
                    self.show_promotion_menu(move)
            else:
                if move in self.board.chess_board.legal_moves:
                    self.board.chess_board.push(move)
                    self.selected_square = None
                    self.check_game_over()
                else:
                    self.selected_square = None

    def is_promotion(self, move: Move) -> bool:
        '''Check if a move is a pawn promotion'''
        piece = self.chess_board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and chess.square_rank(move.to_square) == 7) or \
               (piece.color == chess.BLACK and chess.square_rank(move.to_square) == 0):
                return True
        return False

    def show_promotion_menu(self, move: Move) -> None:
        '''Show a promotion menu using MenuManager from pygamepopup'''
        promotion_menu = InfoBox(
            'Choose a piece to promote to:',
            [
                [Button(title='Knight', callback=lambda: self.promote(
                    move, chess.KNIGHT), size=(TILE_SIZE, TILE_SIZE))],
                [Button(title='Bishop', callback=lambda: self.promote(
                    move, chess.BISHOP), size=(TILE_SIZE, TILE_SIZE))],
                [Button(title='Rook', callback=lambda: self.promote(
                    move, chess.ROOK), size=(TILE_SIZE, TILE_SIZE))],
                [Button(title='Queen', callback=lambda: self.promote(
                    move, chess.QUEEN), size=(TILE_SIZE, TILE_SIZE))]
            ],
            has_close_button=False
        )
        self.menu_manager.open_menu(promotion_menu)

    def promote(self, move: Move, piece: Piece) -> None:
        '''Promotes the pawn to the selected piece and completes the move'''
        promotion_move = chess.Move(move.from_square,
                                    move.to_square, promotion=piece)
        if promotion_move in self.chess_board.legal_moves:
            self.chess_board.push(promotion_move)
        self.menu_manager.close_active_menu()
        self.selected_square = None
        self.check_game_over()

    def check_game_over(self) -> None:
        '''Check if the game is over due to checkmate'''
        if self.chess_board.is_checkmate():
            self.start_new_game()

    def render(self) -> None:
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
        self.menu_manager.display()

    def update(self) -> None:
        self.handle_events()
        self.render()
