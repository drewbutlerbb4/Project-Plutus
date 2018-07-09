from game_generator import GameGenerator
from poker_model import PokerGame


class PokerGenerator(GameGenerator):

    def __init__(self):
        """
        Creates a counter for the number of games made
        """

        self.games_made = 0

    def create_game(self, num_players, hands_per_game):
        """
        Creates a PokerGame with a specific number of players and hands to be played

        :param num_players:     The number of players spaces to be reserved in each game
        :param hands_per_game:  The max number of hands to be played in each game

        :return:    A new instance of a blank Poker Game with the specified number of
                    players and hands per game
        """
        self.games_made += 1
        return PokerGame(num_players, hands_per_game)