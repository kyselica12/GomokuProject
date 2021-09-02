import pandas as pd
import PySimpleGUI as sg
import glob
import numpy as np

class Dictionary:

    def __init__(self):
        df = pd.read_csv("slovnik.csv")
        self.columns = list(df.columns)
        self.data = df.values.tolist()

    def append(self, words):
        sp, sk = values['word'].split(', ')

        if not np.any(np.array(self.data) == sp):
            self.data.append([int(values['lesson']), sp, sk, 0, 0, 0])

    def save(self):
        df = pd.DataFrame(data=self.data, columns=self.columns)
        df.set_index(self.columns[0:2])
        df.to_csv("slovnik.csv", index=False)


class GUI:

    def __init__(self, title, text_input):
        self.title = title
        self.text_input = text_input
        self.none = None

    def action(self):
        pass



class System:

    def __init__(self):
        pass


d = Dictionary()
print(d.columns)

layout = [[sg.Text('Lekcia'), sg.Input(key='lesson')],
          [sg.Text('Zadaj nove slovicko'), sg.Input(tooltip='(spanielske-slovenske)', key='word')],
          [sg.Text('preco to nejde', key='-OUT-', auto_size_text=False)],
          [sg.Button('EXIT'), sg.Button('NEXT')],
          [sg.Button('Submit', visible=False, bind_return_key=True)]]

window = sg.Window('Spanielsky slovnik', layout, icon='spain.ico')

while True:

    event, values = window.read()

    print(event, values)
    if event in ('EXIT',None):
        break
    elif event == 'Submit':
        window['-OUT-'].update(values['word'])
        d.append(values['word'])
        print(d.data)

    window.FindElement('word').Update('')

window.close()
d.save()




