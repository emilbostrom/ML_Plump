import matplotlib.pyplot as plt
from IPython import display

class Plotter:
    def __init__(self):
        plt.ion()

    def plot(self,scores, mean_scores,nr,name):
        plt.figure(nr)
        plt.clf()
        plt.title(f'Training {name}')
        plt.xlabel('Number of Games')
        plt.ylabel('Score')
        if len(scores) > 100:
            plt.plot(scores[-100:])
            plt.plot(mean_scores[-100:])
        else:
            plt.plot(scores)
            plt.plot(mean_scores)
        plt.ylim(ymin=0,ymax=1)
        plt.show(block=False)
        plt.pause(.1)

    def plotInidivualRounds(self,scores,nr,name):
        plt.figure(nr+5)
        plt.clf()
        plt.title(f'Indiviudal rounds')
        plt.xlabel('Round')
        plt.ylabel('Score')
        plt.bar(range(len(scores)),scores)
        plt.ylim(ymin=0,ymax=1)
        plt.show(block=False)
        plt.pause(.1)