from dataclasses import dataclass

import chess
from chess import PieceType, Color
import pygame
from pygame import Surface
from icecream import ic

IMAGES_DIR = 'gui/images'

@dataclass
class Pieces:
    type: PieceType
    color: Color

    def __post_init__(self):
        pass

    def load_pieces(self) -> list[tuple[Surface, Surface]]:
        white_pawn = pygame.image.load(source=f'{IMAGES_DIR}/white_pawn.png')
        white_bishop = pygame.image.load(source=f'{IMAGES_DIR}/white_bishop.png')
        white_knight = pygame.image.load(source=f'{IMAGES_DIR}/white_knight.png')
        white_rook = pygame.image.load(source=f'{IMAGES_DIR}/white_rook.png')
        white_queen = pygame.image.load(source=f'{IMAGES_DIR}/white_queen.png')
        white_king = pygame.image.load(source=f'{IMAGES_DIR}/white_king.png')
        black_pawn = pygame.image.load(source=f'{IMAGES_DIR}/black_pawn.png')
        black_bishop = pygame.image.load(source=f'{IMAGES_DIR}/black_bishop.png')
        black_knight = pygame.image.load(source=f'{IMAGES_DIR}/black_knight.png')
        black_rook = pygame.image.load(source=f'{IMAGES_DIR}/black_rook.png')
        black_queen = pygame.image.load(source=f'{IMAGES_DIR}/black_queen.png')
        black_king = pygame.image.load(source=f'{IMAGES_DIR}/black_king.png')
        pieces = [(white_pawn, black_pawn),
                  (white_bishop, black_bishop),
                  (white_knight, black_knight),
                  (white_rook, black_rook),
                  (white_queen, black_queen),
                  (white_king, black_king)]
        return pieces
    
    def set_position(self, position) -> None:
        self.position = position

    def display_pieces(self):
        pass

        