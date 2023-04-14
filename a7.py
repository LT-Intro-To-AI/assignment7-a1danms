from neural import NeuralNet

or_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [1])]

orn = NeuralNet (2, 3 1)



print("\n\nTraining voter opinion")
print()
voter_opinion = [
    ([.9, .6, .8, .3, .1], [1])
    ([.8, .8, .4, .6, .4], [1])
    ([.7, .2, .4, .6, .3], [1])
    ([.5, .5, .8, .4, .8], [0])
    ([.3, .1, .6, .8, .8], [0])
    ([.6, .3, .4, .3, .6], [0])]

von = NeuralNet(5, 6, 1)

von.train(voter_opinion) 
print(von.test_with_expected(voter_opinion))

