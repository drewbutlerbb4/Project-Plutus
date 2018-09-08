from NEAT.game_models.game_model import GameModel
import random


class XorGame(GameModel):
    """
    An abstract class for handling the trading of information between
    the LearningModel and the Game. The LearningModel will continually
    request information, based on its parameters and GameModel will
    facilitate its learning process by providing data based on the
    Game trying to be learnt
    """

    def __init__(self, num_players, hands_per_game):
        """
        Sets the parameters of the game

        :param num_players:     The number of players participating
        :param hands_per_game:  The number of hands to be played
        """
        self.num_players = num_players
        self.hands_per_game = hands_per_game
        self.cur_input = (0, 0)
        self.cur_hand = 1
        self.correct_answers = 0
        return

    def send_inputs(self):
        """
        Compiles the information that it wants the neural networks to
        interpret and then returns it

        :return:    The number of the neural network to interpret the data
                    and the information to be interpretted by the network
        """
        self.cur_input = [random.randint(0, 1), random.randint(0, 1)]
        return 0, self.cur_input

    def receive_outputs(self, outputs):
        """
        Receives the information about the neural networks outputs and
        uses this to update the state of the game

        :param outputs:     The neural networks interpretation of the
                            the last input set received
        """
        if self.cur_hand <= self.hands_per_game:
            cur_input = self.cur_input
            if (cur_input == (0,0)) | (cur_input == (1, 1)):
                if outputs[0] > outputs[1]:
                    self.correct_answers += 1
            else:
                if outputs[1] > outputs[0]:
                    self.correct_answers += 1
            self.cur_hand += 1
        else:
            return -1

    def send_fitness(self):
        """
        Determines the fitness of each of the neural networks and returns a
        list of the fitness

        :return:    A list of fitness values of each of the neural networks
        """
        if self.cur_hand > self.hands_per_game:
            return [self.correct_answers / self.hands_per_game]
        else:
            return -1