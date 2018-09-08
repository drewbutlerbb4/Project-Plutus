from NEAT.game_generators.game_generator import GameGenerator
from NEAT.game_models.xor_model import XorGame


class XorGenerator(GameGenerator):

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
        return XorGame(num_players, hands_per_game)

    def to_json(self):
        """
        :return: Returns the name of the generator
        """
        return "XorGenerator"
