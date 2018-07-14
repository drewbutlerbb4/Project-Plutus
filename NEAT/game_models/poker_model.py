"""
A class for a PokerGame which extends the abstract class GameModel.
GameModel's are a method to facilitate the growth of neural networks in conjunction
with our learningModel. A GameModel does so by being an intermediary between the
networks and the fitness measurement, that the networks are evaluated on.
PokerGame is a GameModel that connects our learning model to the game of Poker.
Networks will receive information about the current state of the table and will
then return the action they would like to make. After a predetermined number of
hands, the game will stop and the fitness of every player will be available for
the learning model to take
"""
import random
from NEAT.game_models.game_model import GameModel


class PokerGame(GameModel):
    """
    total_hands:    Number of hands to be played before the game is stopped
    num_players:    The number of players involved in the game
    num_actors:     The number of players involved in the hand
    cards_in_deck:  The cards that haven't already been used from the deck
    board:          The cards that are on the board
    hands:          A list of each actors cards
    button_pos:     The actor with the button
    to_act:         The person who currently has the action
    in_action:      A list of whether each actor is still in the action
    player_hand_states: Current amount of money involved from personal stack
    player_money_state: The amount of money that the player has behind
    big_blind:      The value of the big blind
    high_bet:       The value of the current bet
    is_send_state:  Whether the game should be sending or receiving information
    """

    def __init__(self, num_players, hands_per_game):
        if num_players > 24:
            raise NotImplementedError("More than 23 players are not supported")
        self.total_hands = hands_per_game
        self.num_players = num_players
        self.num_actors = num_players
        self.cards_in_deck = [x for x in range(0, 52)]
        self.board = []
        self.hands = []
        self.button_pos = random.randint(0, num_players - 1)
        self.in_action = [True for _ in range(0, num_players)]
        self.player_hand_states = [0 for _ in range(0, num_players)]
        self.player_money_state = [200 for _ in range(0, num_players)]
        self.__deal_hands()
        self.big_blind = 2
        self.high_bet = 2
        self.round_bet = 2
        self.__pay_blinds()
        self.is_send_state = True

    def __pay_blinds(self):
        """
        Posts the big blind and small blind automatically
        """

        pos = self.button_pos + 1
        found_button = False
        while (pos < self.button_pos + 1 + self.num_players) & (not found_button):
            if self.in_action[pos % self.num_players]:
                self.button_pos = pos % self.num_players
                found_button = True
            pos += 1

        small_blind_amount = int(self.big_blind / 2)
        found_small = False
        found_big = False
        found_actor = False
        pos = self.button_pos + 1
        stop_point = pos + self.num_players

        # Finds active players for the positions of big blind, small blind, and in action
        while (pos < stop_point) & (not found_actor):
            if self.in_action[pos % self.num_players]:
                if not found_small:
                    small_blind = pos % self.num_players
                    found_small = True
                elif not found_big:
                    big_blind = pos % self.num_players
                    found_big = True
                elif not found_actor:
                    to_act = pos % self.num_players
                    found_actor = True
            pos += 1

        # Ensures that action starts at the next available player
        if found_actor:
            self.to_act = to_act
        else:
            self.to_act = small_blind

        self.last_to_bet = big_blind

        # Pays up the big and small blind
        if self.player_money_state[big_blind] < self.big_blind:
            self.player_hand_states[big_blind] = self.player_money_state[big_blind]
            self.player_money_state[big_blind] = 0
        else:
            self.player_hand_states[big_blind] = self.big_blind
            self.player_money_state[big_blind] -= self.big_blind

        if self.player_money_state[small_blind] < small_blind_amount:
            self.player_hand_states[small_blind] = self.player_money_state[big_blind]
            self.player_money_state[small_blind] = 0
        else:
            self.player_hand_states[small_blind] = small_blind_amount
            self.player_money_state[small_blind] -= small_blind_amount

    def __deal_hands(self):
        """
        Deals cards to all the players
        """

        # Picks cards for all the players
        for cur_hand in range(0, self.num_players):

            if self.player_money_state[cur_hand] > 0:
                from_deck = random.sample(range(0, len(self.cards_in_deck)), 2)
                from_deck = sorted(from_deck)
                card_1 = self.cards_in_deck.pop(from_deck[0])
                card_2 = self.cards_in_deck.pop(from_deck[1] - 1)
                self.hands.append([card_1, card_2])
            else:
                self.hands.append([])
                self.in_action[cur_hand] = False

        count = 0
        for item in self.player_money_state:
            count += item
        if not count == 800:
            raise NotImplementedError("WHY")

    def send_inputs(self):
        """
        Compiles information about the current game state and returns it

        :return:    Complete information (hopefully) about the game state
                    In its current states, it does not currently return the
                    past hand history and past bet history in this hand
        """

        # Doesn't allow this method if the game is over
        if self.total_hands == -1:
            return -1

        to_act = self.to_act

        card1_value = self.hands[to_act][0] % 13
        card1_suit = int(self.hands[to_act][0] / 13)
        card2_value = self.hands[to_act][1] % 13
        card2_suit = int(self.hands[to_act][1] / 13)

        nn_input = [card1_value, card1_suit, card2_value, card2_suit]

        nn_input.append(self.player_hand_states[to_act])
        nn_input.append(self.player_money_state[to_act])

        # Adds the board cards to the inputs
        for board_card in range(0, 5):
            if (len(self.board) - 1) < board_card:
                nn_input.append(-1)
                nn_input.append(-1)
            else:
                card = self.board[board_card]
                (value, suit) = (card % 13, int(card / 13))
                nn_input.append(value)
                nn_input.append(suit)

        # Adds the other actors states to the input
        for actor in range(to_act + 1, self.num_players):
            nn_input.append(self.player_hand_states[actor])
            nn_input.append(self.player_money_state[actor])
        for actor in range(0, to_act):
            nn_input.append(self.player_hand_states[actor])
            nn_input.append(self.player_money_state[actor])

        self.is_send_state = False

        return self.to_act, nn_input

    def receive_outputs(self, outputs):

        # Doesn't allow this method if the game is over
        if self.total_hands == -1:
            return -1

        if not len(outputs) == 4:
            raise NotImplementedError("Output not recognized")

        call_or_check_value = outputs[0]
        raise_value = outputs[1]
        fold_value = outputs[2]
        action_amount = outputs[3]

        high_value = max(call_or_check_value, raise_value, fold_value)

        # In the event of a call or check
        if high_value == call_or_check_value:
            money_behind = self.player_money_state[self.to_act]
            money_in = self.player_hand_states[self.to_act]

            if not self.high_bet == money_in:
                the_bet = self.player_hand_states[self.last_to_bet]
                if money_behind < (the_bet - money_in):
                    self.player_hand_states[self.to_act] += money_behind
                    self.player_money_state[self.to_act] = 0
                else:
                    self.player_money_state[self.to_act] -= (the_bet - money_in)
                    self.player_hand_states[self.to_act] += (the_bet - money_in)
        # In the event of a raise (check for invalid raises)
        elif high_value == raise_value:
            bet_size = int(action_amount)

            # Forces a raising player to at least bet the size of the last bet
            if bet_size < self.round_bet:
                bet_size = self.round_bet
            # Forces a bet larger than a big blind
            elif bet_size < self.big_blind:
                bet_size = self.big_blind
            # Sets the round bet to the new bet size
            else:
                self.round_bet = bet_size

            # Checks to see if player is betting more than they can
            if bet_size >= self.player_money_state[self.to_act]:
                bet_size = self.player_money_state[self.to_act]

            self.player_hand_states[self.to_act] += bet_size
            self.player_money_state[self.to_act] -= bet_size
            self.high_bet = self.player_hand_states[self.to_act]
            self.last_to_bet = self.to_act
        # In the event of a fold
        else:
            self.in_action[self.to_act] = False
            self.num_actors -= 1

        self.next_actor()
        self.is_send_state = True

    def next_round(self):
        board_state = len(self.board)

        # If the board has 5 cards
        if board_state == 5:
            self.new_hand()
        else:
            # If te board has 0 cards
            if board_state == 0:

                cards = random.sample(range(0, len(self.cards_in_deck)), 3)
                cards = sorted(cards)
                self.board.append(self.cards_in_deck.pop(cards[0]))
                self.board.append(self.cards_in_deck.pop(cards[1] - 1))
                self.board.append(self.cards_in_deck.pop(cards[2] - 2))
            # If the board has 3 or 4 cards
            else:

                card = random.randint(0, len(self.cards_in_deck) - 1)
                self.board.append(self.cards_in_deck.pop(card))

            pos = self.button_pos + 1
            found_actor = False
            while (pos < self.button_pos + 1 + self.num_players) & (not found_actor):
                if self.in_action[pos % self.num_players]:
                    self.to_act = pos % self.num_players
                    found_actor = True
                pos += 1

            self.last_to_bet = self.to_act
            self.high_bet = 0

    def send_fitness(self):
        """
        Returns a list of fitness values if the game is over
        -1 if the game is still going

        :return:    Retrieves the fitness values
        """

        # If the game is over, return the list of fitness values
        if self.total_hands == -1:
            return self.player_money_state
        # If the game is not over, don't return anything
        else:
            return -1

    def next_actor(self):
        """
        Sets up action for the next player to act. If there is no one left to act
        then the hand is evaluated and a new hand is started

        :return:
        """
        cur_actor = self.to_act

        actor_num = cur_actor + 1
        is_chosen = False
        # Checks behind the current actor for active players
        while (not is_chosen) & (actor_num < self.num_players):
            if self.in_action[actor_num]:
                # Only lets actors with money behind act
                if not self.player_money_state[actor_num] == 0:
                    self.to_act = actor_num
                    is_chosen = True
                else:
                    # If the last better has run out of money
                    if self.last_to_bet == actor_num:
                        self.next_round()
                        return
            actor_num += 1

        actor_num = 0
        # Wraps and checks in front of the current actor for active players
        while (not is_chosen) & (actor_num < cur_actor):
            if self.in_action[actor_num]:
                # Only lets actors with money behind act
                if not self.player_money_state[actor_num] == 0:
                    self.to_act = actor_num
                    is_chosen = True
                else:
                    # If the last better has run out of money
                    if self.last_to_bet == actor_num:
                        self.next_round()
                        return
            actor_num += 1

        # If there is only one actor left, or all other players are all in
        if (self.num_actors == 1) | (self.to_act == cur_actor):
            self.new_hand()

        # If this is the last person to bet, move onto the next round
        if self.to_act == self.last_to_bet:
            big_blind_pos = self.button_pos + 2
            if big_blind_pos >= self.num_players:
                big_blind_pos -= self.num_players
            # If this is the big blind 'bet' then ensure the big blind gets a
            # chance to bet
            if ((len(self.board) == 0) and
                    (big_blind_pos == self.to_act)):

                for player_num in range(big_blind_pos + 1, self.num_players):
                    if self.in_action[player_num]:
                        self.last_to_bet = player_num
                        return

                for player_num in range(0, big_blind_pos):
                    if self.in_action[player_num]:
                        self.last_to_bet = player_num
                        return
            # Else move onto the next round
            else:
                self.next_round()

    def new_hand(self):
        """
        Forces the evaluation of the last hand and then resets the game
        to anticipate a new round of play
        """

        self.evaluate_old_hand()
        self.player_hand_states = [0 for _ in range(0, self.num_players)]
        self.total_hands -= 1
        if self.total_hands == -1:
            return
        self.cards_in_deck = [x for x in range(0, 52)]
        self.hands = []
        self.__deal_hands()
        self.board = []
        total_alive = 0
        for players in range(0, self.num_players):
            if self.player_money_state[players] == 0:
                self.in_action[players] = False
            else:
                self.in_action[players] = True
                total_alive += 1
        if total_alive <= 1:
            self.total_hands = -1
            return
        self.num_actors = total_alive
        self.high_bet = 2
        self.__pay_blinds()
        self.round_bet = 2
        if not self.in_action[self.to_act]:
            self.next_actor()
        self.is_send_state = True

    def evaluate_old_hand(self):
        """
        Determines who won the last hand to be played and redistributes
        the chips from the pot
        """

        # If only one actor remains, give them all the chips
        if self.num_actors == 1:
            winner = self.to_act

            pot_size = 0
            for person in range(0, len(self.player_hand_states)):
                pot_size += self.player_hand_states[person]
                self.player_hand_states[person] = 0

            self.player_money_state[winner] += pot_size
        # If there are multiple actors remaining at the end of the hand
        # redistribute pot chips to those who are owed
        else:
            rank = self.determine_rank()

            pot_size = 0
            for chips in self.player_hand_states:
                pot_size += chips

            rank_not_paid = 0

            # Distributes chips in pot, until there are none remaining
            while not pot_size <= 0:
                ranks_owed = [rank_not_paid]
                owed_amounts = [self.player_hand_states[rank[rank_not_paid][2]]]
                rank_not_paid += 1
                tie_checked = False

                # Compiles all of the tied players
                while (not tie_checked) & (rank_not_paid < len(rank)):
                    if rank[rank_not_paid][0] > rank[ranks_owed[(len(ranks_owed)) - 1]][0]:
                        tie_checked = True
                    if self.__cards_are_equal(rank[rank_not_paid],
                                              rank[ranks_owed[len(ranks_owed) - 1]]):

                        ranks_owed.append(rank_not_paid)
                        owed_amounts.append(self.player_hand_states[rank[rank_not_paid][2]])
                        rank_not_paid += 1
                    else:
                        tie_checked = True

                # Sorts the two lists from smallest to largest amount owed
                sorted_list = []
                for actor in range(0, len(ranks_owed)):
                    sorted_list.append((owed_amounts[actor], ranks_owed[actor]))
                sorted_list = sorted(sorted_list)

                ranks_owed = []
                owed_amounts = []
                for actor in range(0, len(sorted_list)):
                    ranks_owed.append(rank[sorted_list[actor][1]][2])
                    owed_amounts.append(sorted_list[actor][0])

                paid_off = self.__payoff_players(ranks_owed, owed_amounts)

                pot_size -= paid_off

    def __payoff_players(self, actors, owed_amounts):
        """
        Distributes money between multiple players who tied for the pot

        :param actors:      List of actors that are tied
        :param owed_amounts:Amount owed to each of the actors
        """

        paid = 0
        total_pot = 0

        # Goes through all the owed players
        while not len(actors) == 0:

            owed = owed_amounts[0]
            pot_size = 0

            # Takes money from all the players to give to the pot
            for person in range(0, len(self.player_hand_states)):
                person_value = self.player_hand_states[person]
                # If the person doesn't have enough to pay all
                if person_value < (owed - paid):
                    pot_size += person_value
                    self.player_hand_states[person] = 0
                else:
                    pot_size += (owed - paid)
                    self.player_hand_states[person] -= (owed - paid)

            divided_pot = int(pot_size / len(actors))
            # Divides the chips to the winners
            for actor in actors:
                self.player_money_state[actor] += divided_pot

            rounding_check = divided_pot * len(actors)
            # Ensures that the money goes somewhere
            while rounding_check < pot_size:
                random_num = random.randint(0, len(actors) - 1)
                self.player_money_state[actors[random_num]] += 1
                rounding_check += 1

            owed_amounts.pop(0)
            actors.pop(0)

            owed_iter = 0
            while owed_iter < len(actors):
                if owed_amounts[owed_iter] == owed:
                    actors.pop(0)
                    owed_amounts.pop(0)
                else:
                    owed_iter = len(actors)

            paid = owed
            total_pot += pot_size

        print(self.player_money_state)
        return total_pot

    def __cards_are_equal(self, card_set1, card_set2):
        """
        Checks that all the cards have the same number. This decides ties
        between ranks, this does not decide the rank of the hand.

        :param card_set1:   A tuple of card rank, list of cards, and actor number
        :param card_set2:   A tuple of card rank, list of cards, and actor number
        :return:            True if the card numbers are equal, False otherwise
        """

        for x in range(0, len(card_set1[1])):
            if not card_set1[1][x][0] == card_set2[1][x][0]:
                return False
        return True

    def determine_rank(self):
        """
        Ranks the active actors in the order of strongest to weakest hands

        :return:    List of active actors in order from strongest to weakest hands
        """

        actor_nums = []
        action_cards = []
        board = self.board

        # Compile the remaining actors
        for actor in range(0, self.num_players):
            if self.in_action[actor]:
                actor_nums.append(actor)

        # Goes through all the active actors and compiles their 7 possible cards
        for actor in actor_nums:
            card1 = (self.hands[actor][0] % 13, int(self.hands[actor][0] / 13))
            card2 = (self.hands[actor][1] % 13, int(self.hands[actor][1] / 13))
            cards = [card1, card2]
            for card in board:
                cur_card = (card % 13, int(card / 13))
                cards.append(cur_card)
            action_cards.append(self.determine_strength(cards))

        tiebreak_iter = 0

        # Sorts two lists by rank
        sorter = []
        for sort_iter in range(0, len(action_cards)):
            sorter.append((action_cards[sort_iter][0], (action_cards[sort_iter][1], actor_nums[sort_iter])))

        sorter = sorted(sorter)

        best_actors = []
        best_cards = []

        for item in sorter:
            best_actors.append(item[1][1])
            best_cards.append((item[0], item[1][0]))

        # Decides tie breakers in hand rankings
        while tiebreak_iter < len(best_actors) - 1:

            tie_value = best_cards[tiebreak_iter][0]
            tie_start = tiebreak_iter
            tie_end_help = tiebreak_iter
            tie_end = tiebreak_iter
            # Finds how large the list of ties is
            while tie_end_help < len(best_actors):
                if best_cards[tie_end_help][0] == tie_value:
                    tie_end += 1
                    tie_end_help += 1
                else:
                    tie_end_help = len(best_actors)
            tie_end -= 1

            # Sorts through this list of ties
            for end in range(tie_end, tie_start, -1):
                for begin in range(tie_start, end):

                    card_set1 = best_cards[begin]
                    card_set2 = best_cards[begin + 1]

                    # Switch elements if they are in the incorrect order
                    if not self.is_card_set1_stronger(card_set1, card_set2):
                        temp = best_actors[begin]
                        best_actors[begin] = best_actors[begin + 1]
                        best_actors[begin + 1] = temp
                        best_cards[begin] = best_cards[begin + 1]
                        best_cards[begin + 1] = card_set1
            tiebreak_iter = tie_end + 1

        # DEBUGGING
        print("HAND RESULTs")
        string1 = "                  ["
        for item in self.hands:
            if not item == []:
                string1 += "(" + str(item[0] % 13) + "," + str(int(item[0]/13))
                string1 += "),(" + str(item[1] % 13) + "," + str(int(item[1] / 13)) + "), "
            else:
                string1 += "(X,X),(X,X),"
        string1 += "] HANDS"
        print(string1)
        string1 = "                  ["
        for item in self.board:
            string1 += "(" + str(item % 13) + "," + str(int(item / 13)) + "), "
        string1 += "] BOARD"
        print(string1)
        print(best_actors)
        print(best_cards)
        print("END RESULTS")

        ranks = []

        for actors in range(0, len(best_cards)):
            ranks.append((best_cards[actors][0], best_cards[actors][1], best_actors[actors]))

        return ranks

    def is_card_set1_stronger(self, card_set1, card_set2):
        """
        Returns if card_set1 is stronger or not (in the event of a tie)

        :param card_set1:   The card set of player1
        :param card_set2:   The card set of player2
        :return:            True if player1 is stronger, False otherwise
        """

        for card in range(0, len(card_set1[1])):
            if card_set1[1][card][0] > card_set2[1][card][0]:
                return True
            elif card_set1[1][card][0] < card_set2[1][card][0]:
                return False
        return False

    def determine_strength(self, cards):
        """
        Decides the strongest hand and then returns the value of that hand
        (explained below) and the hand itself
        (Value: Straight-Flush=1, Four of a Kind=2, Full House=3, Flush=4,
        Straight=5, Three of a kind=6, Two pair=7, Pair=8, High card=9)

        :param cards:   The cards that can be used
        :return:        The strongest hand and the value of that hand
        """

        cards = sorted(cards)
        straight = []
        match = []
        flush = []
        straight_count = 1
        cur_match_count = 0
        flush_count = 0
        aces_exist = False
        aces = []
        ace_iter = -1

        # Compiles aces for wrapping straights
        while cards[ace_iter][0] == 12:
            aces_exist = True
            aces.append(cards[ace_iter])
            ace_iter -= 1

        last_card = -2
        # TODO More elegant solution may be to not bum the straight_count on a match
        # TODO just dont set it to 0. (Low importance)
        # Iterates over the cards sorted by number looking for matches and straights
        for card_num in range(0, len(cards)):
            card = cards[card_num]

            # If the card matches the last card then we add to our match count
            if last_card == card[0]:
                cur_match_count += 1
                straight_count += 1
            # If the card continues the straight then add to our straight count
            elif (last_card + 1) == card[0]:
                # In case of a match between a straight, add the match to pairs
                if cur_match_count > 0:
                    one_match = []
                    for match_iter in range(card_num - cur_match_count - 1, card_num):
                        one_match.append(cards[match_iter])
                    match.append(one_match)
                straight_count += 1
                cur_match_count = 0
            # Else, check for matches and straights to save
            else:
                # If there is a match, save it
                if cur_match_count > 0:
                    one_match = []
                    for match_iter in range(card_num - cur_match_count - 1, card_num):
                        one_match.append(cards[match_iter])
                    match.append(one_match)
                # Checks for an Ace to 5 straight
                elif (straight_count == 4) & aces_exist:
                    if (card_num == 4) & (cards[0][0] == 0) & (cards[1][0] == 1) & (
                            cards[2][0] == 2) & (cards[3][0] == 3):
                        for straight_check in range(0, 4):
                            straight.append(cards[straight_check])
                        straight.extend(aces)
                # If there is a straight, then save it
                elif straight_count >= 5:
                    straight = []
                    for straight_iter in range(card_num - straight_count, card_num):
                        straight.append(cards[straight_iter])
                    if aces_exist & (straight[0][0] == 0):
                        straight.extend(aces)
                straight_count = 1
                cur_match_count = 0
            last_card = card[0]

        # Checks and saves any straights or matches that go to the end of the list
        if cur_match_count > 0:
            one_match = []
            for match_iter in range(len(cards) - cur_match_count - 1, len(cards)):
                one_match.append(cards[match_iter])
            match.append(one_match)
        if straight_count >= 5:
            straight = []
            for straight_iter in range(len(cards) - straight_count, len(cards)):
                straight.append(cards[straight_iter])
            if aces_exist & (straight[0][0] == 0):
                straight.extend(aces)

        # Purge straight collections of duplicate numbers
        # Do so by taking the duplicate number with the most average suit
        if straight:
            suits = [0, 0, 0, 0]
            for card in straight:
                suits[card[1]] += 1
            avg_suit = 0
            for suit_num in range(1, 4):
                if suits[suit_num] > suits[avg_suit]:
                    avg_suit = suit_num

            purged_straight = [straight[0]]
            for card_num in range(1, len(straight)):
                if purged_straight[-1][0] == straight[card_num][0]:
                    if straight[card_num][1] == avg_suit:
                        purged_straight[-1] = straight[card_num]
                else:
                    purged_straight.append(straight[card_num])
            straight = purged_straight
            # Ensures that the straight is still of size 5
            if len(straight) < 5:
                straight = []

        cards = sorted(cards, key=lambda x: x[1])

        last_card = -2
        # Iterates over the cards sorted by suit looking for flushes
        for card_num in range(0, len(cards)):
            card = cards[card_num]

            if last_card == card[1]:
                flush_count += 1
            else:
                if flush_count >= 5:
                    for flush_iter in range(card_num - flush_count, card_num):
                        flush.append(cards[flush_iter])
                flush_count = 1
            last_card = card[1]

        if flush_count >= 5:
            for flush_iter in range(len(cards) - flush_count, len(cards)):
                flush.append(cards[flush_iter])

        # If there is a flush
        if flush:
            # If there is a straight
            if straight:
                shared = []
                for (val1, suit1) in straight:
                    for (val2, suit2) in flush:
                        if (val1 == val2) & (suit1 == suit2):
                            shared.append((val1, suit1))

                # Check for straight flush
                if len(shared) >= 5:

                    aces_exist = False
                    if shared[-1][0] == 12:
                        aces_exist = True
                    last_card = shared[0]
                    straight_count = 1
                    # Iterates through the shared cards looking for a straight
                    for card_num in range(1, len(shared)):
                        if (last_card[0] + 1) == shared[card_num][0]:
                            straight_count += 1
                        else:
                            if straight_count >= 5:
                                return 1, shared[card_num - 5: card_num]
                            elif (straight_count == 4) & aces_exist:
                                to_return = shared[card_num - 4: card_num]
                                to_return.append(shared[-1])
                                return 1, to_return
                            straight_count = 1
                        last_card = shared[card_num]
                    if straight_count >= 5:
                        return 1, shared[len(shared) - 5:]

                # Else check for four of a kind and full house
                best_pairs = self.biggest_pairs(cards, match)
                if best_pairs[0] < 4:
                    return best_pairs
                else:
                    return 4, sorted(flush[len(flush) - 5:], reverse=True)
            else:
                best_pairs = self.biggest_pairs(cards, match)
                # If there is a four of a kind or full house, use that
                if best_pairs[0] < 4:
                    return best_pairs
                # Else use the straight
                else:
                    return 4, sorted(flush[len(flush) - 5:], reverse=True)
        else:
            # If there is a straight, check for a four of a kind or full house
            if straight:
                best_pairs = self.biggest_pairs(cards, match)
                if best_pairs[0] < 4:
                    return best_pairs
                else:
                    return 5, straight[len(straight) - 5:]
            else:
                return self.biggest_pairs(cards, match)

    def biggest_pairs(self, cards, pairs):
        """
        Given the cards and pairings, find the value and cards of the highest
        value of combination of cards

        :param cards:   The cards that are playable
        :param pairs:   A set of 2, 3, and 4 pairs in the playable cards set
        :return:        The value and cards of the highest value combination of cards
        """
        two_pair = []
        three_pair = []

        cards = sorted(cards)

        # Divides the pairs into 4, 3, and 2 pairs
        for pair in pairs:
            if len(pair) == 4:
                # Remove the 4 of a kind cards, so they can't be chosen as the high card
                for item in pair:
                    cards.remove(item)
                pair.append(cards.pop(-1))
                return 2, pair
            elif len(pair) == 3:
                three_pair.append(pair)
            else:
                two_pair.append(pair)

        if len(three_pair) > 0:
            # If there is substance for a full house, find the largest full house
            if (len(three_pair) > 1) | (len(two_pair) > 0):
                house_three = three_pair[0]
                house_two = [[-1]]
                # Checks for the highest 3 of a kind
                for three_iter in range(1, len(three_pair)):
                    if house_three[0][0] < three_pair[three_iter][0][0]:
                        house_three = three_pair[three_iter]

                # Checks for the second highest 3 of a kind for the pair in the full house
                for three_iter in range(0, len(three_pair)):
                    if house_two[0][0] < three_pair[three_iter][0][0]:
                        if not house_three[0][0] == three_pair[three_iter][0][0]:
                            house_two = three_pair[three_iter][0:2]

                # Checks for the highest pair to see if it is better for the full house
                for two_iter in range(0, len(two_pair)):
                    if house_two[0][0] < two_pair[two_iter][0][0]:
                        house_two = two_pair[two_iter]

                house_three.append(house_two.pop(0))
                house_three.append(house_two.pop(0))
                return 3, house_three
            # If there is a three pair, return it
            else:
                for item in three_pair[0]:
                    cards.remove(item)
                three_pair[0].append(cards.pop(-1))
                three_pair[0].append(cards.pop(-1))
                return 6, three_pair[0]
        elif len(two_pair) > 0:
            # Return a two pair (two sets of two matching numbers)
            to_return = []
            if len(two_pair) == 3:
                min_pair = min([two_pair[0][0][0], two_pair[1][0][0], two_pair[2][0][0]])
                if not min_pair == two_pair[2][0][0]:
                    to_return.extend(two_pair[2])
                if not min_pair == two_pair[1][0][0]:
                    to_return.extend(two_pair[1])
                if not min_pair == two_pair[0][0][0]:
                    to_return.extend(two_pair[0])

                for item in to_return:
                    cards.remove(item)
                to_return.append(cards.pop(-1))
                return 7, to_return
            # Return a two pair (two sets of two matching numbers
            elif len(two_pair) == 2:
                to_return.extend(two_pair[1])
                to_return.extend(two_pair[0])
                for item in to_return:
                    cards.remove(item)
                to_return.append(cards.pop(-1))
                return 7, to_return
            # Returns a pair
            else:
                to_return.extend(two_pair[0])
                for item in to_return:
                    cards.remove(item)
                cards = sorted(cards)
                to_return.append(cards.pop(-1))
                to_return.append(cards.pop(-1))
                to_return.append(cards.pop(-1))
                return 8, to_return
        # Returns a high card
        else:
            to_return = []
            cards = sorted(cards)
            for x in range(0, 5):
                to_return.append(cards.pop(-1))
            return 9, to_return
