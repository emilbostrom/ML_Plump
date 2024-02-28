import torch
import random
import numpy as np 
from collections import deque
import re
import itertools
from modelGuess import Linear_QNet, QTrainer
from helper import Plotter

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class GuessAgent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # model, trainer
        self.model = Linear_QNet(72,100,100,50,11) # for training
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        #self.model.load_state_dict(torch.load('./model/model2.pth'))
        #self.model.eval()
        self.plotter = Plotter()

    def get_state(self,inputState):
        '''
        Position at the table (5)
        Number of cards guessed (15)
        Cards in hand (52)
        '''
        posAtTable = []
        for pos in range(1,6):
            if pos == inputState[-1]:
                posAtTable.append(1)
            else:
                posAtTable.append(0)
        inputState.pop(-1)

        sumOfGuesses = []
        for guess in range(1,16):
            if guess == inputState[-1]:
                sumOfGuesses.append(1)
            else:
                sumOfGuesses.append(0)
        inputState.pop(-1)

        cards = []
        for card in range(1,53):
            if card in inputState:
                cards.append(1)
            else:
                cards.append(0)    

        if sum(cards) == 10 and posAtTable[4]==1:
            done = True
        else:
            done = False

        state = [
            posAtTable,
            sumOfGuesses,
            cards
        ]

        state = list(itertools.chain(*state))
        
        return np.array(state,dtype=int), done

    def remember(self, state,next_state,cardGuess,reward,done):
        self.memory.append((state,next_state,cardGuess,reward,done)) # popleft if max memory reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states,next_states,cardGuesss,rewards,dones = zip(*mini_sample)
        self.trainer.train_step(states,next_states,cardGuesss,rewards,dones)

    def train_short_memory(self,state,next_state,cardGuess,reward,done):
        self.trainer.train_step(state,next_state,cardGuess,reward,done)

    def get_action(self,state,legalMoves):
        '''
        Output: Cards to take guess (11) 
        '''
        self.epsilon = np.exp(-self.n_games*0.01)
        cardGuess = [0,0,0,0,0,0,0,0,0,0,0]

        if np.random.rand(1) < self.epsilon:
            card = random.choice(legalMoves) - 1
            cardGuess[card] = 1
            return cardGuess.index(1)
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)

        sorted_prediction = sorted(prediction, reverse=True)

        for i, value in enumerate(sorted_prediction): # loop through and find best legal move
            guesses = ((prediction == value).nonzero(as_tuple=True)[0])
            for guess in guesses:
                if guess in legalMoves:
                    cardGuess[guess] = 1
                    return cardGuess.index(1)
    
    def get_reward(self,cardGuess,dataGuess,state):
        print(dataGuess)
        difference = np.abs(cardGuess-int(dataGuess))

        if difference == 0:
            reward = 10
        elif difference == 1:
            reward = 0
        else:
            reward = -10

        return reward


def train():
    plot_score = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    score = 0
    agent = GuessAgent()

    train = open("guessTraining.txt", "r")
    results = open("guessResults.txt", "r")
    for count,line in enumerate(train):
    
        # get current state
        inputState = [int(s) for s in re.findall(r'\b\d+\b', line)]
        next_state_input = inputState.copy()
        
        state_curr,done = agent.get_state(inputState)

        legalMoves = [0,1,2,3,4,5,6,7,8,9,10] 
        if state_curr[0] == 5:
            forbiddenGuess = len(next_state_input) - state_curr[1]
            if forbiddenGuess in legalMoves:
                legalMoves.remove(forbiddenGuess)

        # get move
        cardGuess= agent.get_action(state_curr,legalMoves)

        # next_state after action
        next_state_input[-2] = cardGuess + next_state_input[-2]
        next_state,done = agent.get_state(next_state_input)

        
        # get reward based on data
        reward = agent.get_reward(cardGuess,results.readline(),state_curr)

        score += reward

        # train short memory
        agent.train_short_memory(state_curr,next_state,[cardGuess],[reward],[done])

        # remember
        agent.remember(state_curr,next_state,[cardGuess],[reward],[done])

        if done:
            agent.train_long_memory()
            agent.n_games += 1

            if score > record:
                record = score
                agent.model.save(f'Guesser_hardcode.pth')

            print('Game:',agent.n_games,'Score',score,'Record',record)

            plot_score.append(score)
            total_score += score
            mean_score = total_score/ agent.n_games
            plot_mean_scores.append(mean_score)
            agent.plotter.plot(plot_score,plot_mean_scores,1,'guesser')

            score = 0

    train.close()
    results.close()

if __name__ == "__main__":
    train()