import random

import tqdm

from game import Game, GameState
from play.player import Player


def play_game(player1: Player, player2: Player, game: Game, show=False, random_start=False):

    state = game.get_new_state()

    if random_start:
        move = (random.randrange(game.size), random.randrange(game.size))
        state = game.move(state, move)

        available_moves = game.available_moves(state)
        move = available_moves[random.randrange(len(available_moves))]
        state = game.move(state, move)

    if show:
        print(state)

    while not state.terminal:
        if state.on_turn == 1:
            move = player1.move(state)
        else:
            move = player2.move(state)

        state = game.move(state, move)

        if show:
            print(state)

    return state


def play_games(player1, player2, game, n_games, random_start=False):

    p1, p2 = player1, player2
    p1_name, p2_name = 'p1', 'p2'

    final_states = []
    score = {p1_name:0, p2_name:0, 'draw':0}
    for i in tqdm.tqdm(range(n_games), desc="Playing Games "):
        state = play_game(p1, p2, game, show=False, random_start=random_start)
        final_states.append(state)

        if state.reward == 0:
            score['draw'] += 1
        else:
            winner = -state.on_turn
            if winner == 1:
                score[p1_name] += 1
            else:
                score[p2_name] += 1

        p1, p2 = p2, p1
        p1_name, p2_name = p2_name, p1_name

    return final_states, score


