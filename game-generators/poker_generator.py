from game_generator import GameGenerator
from poker_model import PokerGame


class PokerGenerator(GameGenerator):

    def __init__(self, num_players, hands_per_game):
        """
        Saves the type of game that are to be generated

        :param num_players:     The number of players spaces to be reserved in each game
        :param hands_per_game:  The max number of hands to be played in each game
        """
        self.num_players = num_players
        self.hands_per_game = hands_per_game

    def create_game(self):
        """
        :return:    A new instance of a blank Poker Game with the specified number of
                    players and hands per game
        """
        return PokerGame(self.num_players, self.hands_per_game)