import os
import sys

import pygame

from gui.interface import Interface
from constants import SCREEN_SIZE

os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['SDL_VIDEO_CENTERED'] = '1'  # Centre display window.

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption('Chess')
    interface = Interface(screen, None)
    clock = pygame.time.Clock()
    while True:
        interface.handle_events()  
        interface.render()  
        pygame.display.flip()
        clock.tick(60)

if __name__ == '__main__':
    main()
