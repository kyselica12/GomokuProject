from game import GameState
from player.player import Player

class HumanPlayer(Player):

    def move(self, state: GameState):
        moves = self.game.available_moves(state)
        move = self._get_input(moves)
        return move

    def _get_input(self, moves):
        print("Human player move: ")
        row = int(input('\tRow   : '))
        col = int(input('\tColumn: '))
        move = (row, col)
        if move not in moves:
            print("Bad move! Try again.\n")
            return self._get_input(moves)

        return move