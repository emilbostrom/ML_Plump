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

    def __init__(self,name):
        self.plot_score = []
        self.plot_mean_scores = []
        self.total_score = 0
        self.record = 0
        self.score = 0
        self.win = 0
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.round = deque(maxlen=MAX_MEMORY)
        # model, trainer
        self.model = Linear_QNet(72,100,100,50,11) # for training
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        self.model.load_state_dict(torch.load(f'./model/Guesser_hardcode.pth'))
        self.model.eval()
        self.plotter = Plotter()
        self.training = False

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

        if sum(cards) == 2:
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

        if np.random.rand(1) < self.epsilon and self.training:
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

    def trainMove(self,inputState,legalMoves):

        self.next_state_input = inputState.copy()
        
        self.state_curr,done = self.get_state(inputState)

        # get move
        self.cardGuess= self.get_action(self.state_curr,legalMoves)

        # next_state after action
        self.next_state_input[-2] = self.cardGuess + self.next_state_input[-2]
        self.next_state,done = self.get_state(self.next_state_input)

        self.round.append((self.state_curr,self.next_state,[self.cardGuess],[done]))
        print(self.cardGuess)
        return self.cardGuess

    def trainSave(self,reward,name,game_over):        
        for roundNr in range(len(self.round)):
            state,next_state,cardGuess,done = zip(self.round[roundNr])
    
            self.score += reward

            # train short memory
            self.train_short_memory(state,next_state,cardGuess,[reward],done)

            # remember
            self.remember(state,next_state,cardGuess,[reward],done)

        if reward != 0:
            self.win += 1

        # delete all content from saved rounds
        self.round.clear()
        if game_over:
            self.train_long_memory()
            self.n_games += 1

            if self.score > self.record:
                self.record = self.score
                self.model.save(f'Guesser_{name}.pth')

            print('Game:',self.n_games,'Score',self.score,'Record',self.record)

            self.score = 0
            self.win = 0

