import os
import sys

import pygame

from gui.interface import Interface
from constants import SCREEN_SIZE

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption('Chess')
    interface = Interface(screen)
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        interface.render()
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
