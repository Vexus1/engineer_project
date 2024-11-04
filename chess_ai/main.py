import os

import pygame

from gui.board import Board
from gui.pieces import Pieces

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

if __name__ == '__main__':
    main()