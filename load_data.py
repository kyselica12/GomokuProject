from collections import defaultdict

import numpy as np
import glob
from game import *
import tqdm

PATH = "C:\\Users\\user\\Downloads\\gomocup2019results"

folders = [path for path in glob.iglob(f"{PATH}/*")]

def terminal_game(path):
    size = 15
    count = 0
    terminal = False
    winner = 0
    with open(path, 'r') as game:
        line = game.readline()
        if "20x20" in line:
            size = 20
        elif "15x15" in line:
            size = 15
        else:
            size = -1

        moves = []
        line = game.readline()
        while line[0].isdigit():
            col, row, _ = line.split(',')
            moves.append((int(row)-1, int(col)-1))
            count += 1
            line = game.readline()
        if len(moves) %2 == 0:
            on_turn = 1
        else:
            on_turn = -1

        state = GameState(moves=tuple(moves[:-1]), on_turn=-on_turn, size=size)
        if len(moves) > 0:
            game = Game(size=size)
            state = game.move(state, moves[-1])

        if state.terminal:
            return size, state

    return 0, None

def load_data(path):
    games = defaultdict(list)
    for folder in folders:
        print(folder)
        for path in glob.iglob(f"{folder}/*.psq"):
            size, state = terminal_game(path)
            if state is not None:
                games[size].append(state)

    for size in games:
        all_moves = defaultdict(list)
        for i in tqdm.tqdm(range(len(games[size]))):
            moves = list(map(list, games[size][i].moves))
            all_moves[games[size][i].reward].append(moves)
        for key, moves in all_moves.items():
            winner = 'draw'
            if key == -1:
                winner = 'X'
            elif key == 1:
                winner = "O"
            else:
                winner = "D"

            json_string = json.dumps(moves)
            with open(f"C:\\Users\\user\\Documents\\GomokuProject\\games_{size}x{size}/games_{winner}.json", 'w') as j:
                print(json_string, file=j)

    print(f"saved games: {sum([len(x) for x in games])}")
    for s in games:
        print(f'\tSize {s} -> {len(games[s])} games')

if __name__ == "__main__":
    import sys

    n = len(sys.argv)
    if n == 1:
        print('Missing input path!')
    elif n > 2:
        print("Too many arguments, expecting one!")
    else:
        path = sys.argv[1]
        load_data(path)

