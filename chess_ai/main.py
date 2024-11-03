from kivy.app import App

from gui import ChessBoard

class Chess(App):
    def build(self):
        return ChessBoard()
    

if __name__ == '__main__':
    Chess().run()
