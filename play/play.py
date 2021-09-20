
from game import Game, GameState
from play.player import Player


def play_game(player1: Player, player2: Player, game: Game, show=False):

    state = game.get_new_state()

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


def play_games(player1, player2, game, n_games):

    p1, p2 = player1, player2
    p1_name, p2_name = 'p1', 'p2'

    final_states = []
    score = {p1_name:0, p2_name:0, 'draw':0}
    for i in range(n_games):
        state = play_game(p1, p2, game, show=False)
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


