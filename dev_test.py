from poker_model import PokerGame

game = PokerGame(4, 10000)

"""
print(game.determine_strength([(1,0), (11,0), (10,3), (2,3), (4,0),(3,0),(3,1)]))
print("TIME")


"""

x = 1
while not (x == -1):
    #print(game.send_inputs(),end="INPUTS\n")
    # print(game.receive_outputs([1,0,0,0]))
    game.send_inputs()
    x = game.receive_outputs([1,0,0,0])
    # print(x)

print(game.send_fitness())
