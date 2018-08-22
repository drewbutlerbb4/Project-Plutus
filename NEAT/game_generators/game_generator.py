from abc import ABC, abstractmethod


class GameGenerator(ABC):
    """
    An abstract class for creating the handler of the trading of information between
    the LearningModel and the Game. The LearningModel will be given a GameGenerator
    and will be allowed to request as many Games as they need. The LearningModel will
    then trade information with each individual Game.
    """

    @abstractmethod
    def __init__(self, num_players, hands_per_game):
        """
        Sets the parameters of the game

        :param num_players:     The number of players participating
        :param hands_per_game:  The number of hands to be played
        """
        pass

    @abstractmethod
    def create_game(self):
        """
        Creates and returns a new instance of the game the class
        is modeled for

        :return:    Returns an instance of the game that this class is a generator of
        """
        pass

    @abstractmethod
    def to_json(self):
        """
        :return: Returns the name of the game generator
        """
        pass