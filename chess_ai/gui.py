from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle
from kivy.config import Config
from kivy.clock import Clock
import chess
import threading
from chess import Piece

Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

class ChessBoard(GridLayout):
    def __init__(self, **kwargs):
        super().__init__(cols=8, **kwargs)
        self.board = chess.Board()
        self.selected_square = None  
        self.buttons = []
        self.create_buttons()
        self.update_board()

    def create_buttons(self) -> None:
        for _ in range(64):
            button = BoxLayout()
            with button.canvas.before:
                button.bg_color = Color(1, 1, 1, 1)  
                button.rect = Rectangle(size=button.size, pos=button.pos)
            button.bind(pos=self.update_rect, size=self.update_rect)
            self.add_widget(button)
            self.buttons.append(button)
            button.bind(on_touch_down=self.on_button_press)

    def update_rect(self, instance, value) -> None:
        instance.rect.pos = instance.pos
        instance.rect.size = instance.size

    def on_button_press(self, instance, touch) -> None:
        if instance.collide_point(*touch.pos):
            index = self.buttons.index(instance)
            move = self.get_move_from_input(index)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.update_board() 
                if not self.board.is_game_over():
                    threading.Thread(target=self.ai_move).start()  

    def get_move_from_input(self, index: int) -> None:
        reversed_index = 63 - index 
        square = chess.SQUARE_NAMES[reversed_index]
        if self.selected_square is None:
            piece = self.board.piece_at(reversed_index)
            if piece is not None and piece.color == self.board.turn:
                self.selected_square = square
            return None
        else:
            move = chess.Move.from_uci(self.selected_square + square)
            self.selected_square = None 
            return move

    def update_board(self) -> None:
        for i, button in enumerate(self.buttons):
            button.clear_widgets()  
            reversed_index = 63 - i  
            piece = self.board.piece_at(reversed_index)
            row = i // 8
            col = i % 8
            if (row + col) % 2 == 0:
                button.bg_color.rgba = (1, 1, 1, 1)  
            else:
                button.bg_color.rgba = (0.7, 0.7, 0.7, 1) 
            if piece:
                piece_name = self.get_piece_image_name(piece)
                image = Image(source=f'images/{piece_name}.png', allow_stretch=True, keep_ratio=True)
                button.add_widget(image)

    def get_piece_image_name(self, piece: Piece) -> str:
        if piece.color == chess.WHITE:
            color = 'white' 
        else:
            color = 'black'
        piece_type = {
            chess.PAWN: 'pawn',
            chess.KNIGHT: 'knight',
            chess.BISHOP: 'bishop',
            chess.ROOK: 'rook',
            chess.QUEEN: 'queen',
            chess.KING: 'king'
        }[piece.piece_type]
        return f'{color}_{piece_type}'
