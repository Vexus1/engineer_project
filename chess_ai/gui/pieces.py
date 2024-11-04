from dataclasses import dataclass

import chess
from chess import PieceType, Color
import pygame
from pygame import Surface
from icecream import ic

from constants import TILE_SIZE

IMAGES_DIR = 'gui/images'

@dataclass
class Pieces:
    type: PieceType
    color: Color

    def __post_init__(self):
        pass

    def load_piece_images(self) -> dict[str, Surface]:
        pieces_image = {
            'P': pygame.image.load(f'{IMAGES_DIR}/white_pawn.png'),    
            'N': pygame.image.load(f'{IMAGES_DIR}/white_knight.png'), 
            'B': pygame.image.load(f'{IMAGES_DIR}/white_knight.png'),
            'R': pygame.image.load(f'{IMAGES_DIR}/white_rook.png'),
            'Q': pygame.image.load(f'{IMAGES_DIR}/white_queen.png'),
            'K': pygame.image.load(f'{IMAGES_DIR}/white_king.png'),
            'p': pygame.image.load(f'{IMAGES_DIR}/black_pawn.png'),
            'n': pygame.image.load(f'{IMAGES_DIR}/black_knight.png'),
            'b': pygame.image.load(f'{IMAGES_DIR}/black_bishop.png'),
            'r': pygame.image.load(f'{IMAGES_DIR}/black_rook.png'),
            'q': pygame.image.load(f'{IMAGES_DIR}/black_queen.png'),
            'k': pygame.image.load(f'{IMAGES_DIR}/black_king.png')
        }
        return pieces_image
    
    def scale_piece_images(self, piece_images: dict[str, Surface]) -> dict[str, Surface]:
        scaled_images = {}
        for piece, image in piece_images.items():
            scaled_images[piece] = pygame.transform.scale(image, (TILE_SIZE,
                                                                  TILE_SIZE))
        return scaled_images

    def set_position(self, position) -> None:
        self.position = position

    def display_pieces(self):
        pass

        