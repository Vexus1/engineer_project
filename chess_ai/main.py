from kivy.app import App

from gui import interface

class Chess(App):
    def build(self):
        return interface
    

if __name__ == '__main__':
    Chess().run()
