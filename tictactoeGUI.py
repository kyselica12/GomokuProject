import PySimpleGUI as sg
from game import *
import numpy as np

SIZE = 15



layout = [[sg.Text('Piškvorky', key='TEXT',auto_size_text=False)]]
layout += [[sg.Button('', key=(i,j), button_color=('white', 'white'), size=(5,2), pad=(0,0)) for j in range(SIZE)] for i in range(SIZE)]



window = sg.Window('Piškvorky', layout)

game = Game()
s = game.get_new_state()

history = []

while True:

    event, values = window.read()

    print(event, values)
    
    if event in ('EXIT',None):
        break
    
    if isinstance(event, tuple):
        move = event
        if move in game.available_moves(s):
            p = 'X' if s.on_turn == 1 else 'O'
            c = 'red' if p == 'X' else 'green'
            window[event].update(text=p, button_color=(c, 'white'))
            s = game.move(s, move)

            history.append(move)
            if s.terminal:
                window['TEXT'].update(f'Vyhral hrac {p}')

from time import time      
np.savetxt(f'game{time()}.history', np.array(history, dtype=int))
    

window.close()

