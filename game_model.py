from abc import ABC, abstractmethod


class GameModel(ABC):
    """
    An abstract class for handling the trading of information between
    the LearningModel and the Game. The LearningModel will continually
    request information, based on its parameters and GameModel will
    facilitate its learning process by providing data based on the
    Game trying to be learnt
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
    def send_inputs(self):
        """
        Compiles the information that it wants the neural networks to
        interpret and then returns it

        :return:    The number of the neural network to interpret the data
                    and the information to be interpretted by the network
        """
        pass

    @abstractmethod
    def receive_outputs(self, outputs):
        """
        Receives the information about the neural networks outputs and
        uses this to update the state of the game

        :param outputs:     The neural networks interpretation of the
                            the last input set received
        """
        pass

    @abstractmethod
    def send_fitness(self):
        """
        Determines the fitness of each of the neural networks and returns a
        list of the fitness

        :return:    A list of fitness values of each of the neural networks
        """
        pass