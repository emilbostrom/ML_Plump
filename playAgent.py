import torch
import random
import numpy as np 
from collections import deque
import re
import itertools
from modelPlay import Linear_QNet, QTrainer
from helper import Plotter
from operator import add

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0005

class PlayAgent:

    def __init__(self,name):
        self.plot_score = []
        self.plot_mean_scores = []
        self.total_score = 0
        self.record = 0
        self.score = 0
        self.round_score = [] 
        self.round_score_tot = [0]*17
        self.win = 0
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 1.0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.round = deque(maxlen=MAX_MEMORY)
        # model, trainer
        self.model = Linear_QNet(194,200,200,100,52) # for training
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        self.model.load_state_dict(torch.load(f'./model/Player_1.pth'))
        self.model.eval()
        self.plotter = Plotter()
        self.training = False 


    def get_state(self,inputState):
        '''
        Cards in hand (52)
        Cards on table (52)
        Cards that have been played (52)
        Cards guessed by player (11) 0-10
        Cards taken by player (11) 0-10
        Total cards guessed players (16) 0-15
        '''
        cards = []
        cardsOnTable = []
        cardsPlayed = []
        cardGuess = []
        cardsTaken = []
        sumOfGuesses = []
        
        cardsInt = inputState[0]
        for cardIdx in range(1,53):
            if cardIdx in cardsInt:
                cards.append(1)
            else:
                cards.append(0)
        
        cardsOnTableInt = inputState[1]
        for cardIdx in range(1,53):
            if cardIdx in cardsOnTableInt:
                cardsOnTable.append(1)
            else:
                cardsOnTable.append(0)

        cardsPlayedInt = inputState[2]
        for cardIdx in range(1,53):
            if cardIdx in cardsPlayedInt:
                cardsPlayed.append(1)
            else:
                cardsPlayed.append(0)

        cardGuessInt = inputState[3]
        for guessIdx in range(1,12):
            if guessIdx in cardGuessInt:
                cardGuess.append(1)
            else:
                cardGuess.append(0)

        cardsTakenInt = inputState[4]
        for takenIdx in range(1,12):
            if takenIdx in cardsTakenInt:
                cardsTaken.append(1)
            else:
                cardsTaken.append(0)

        sumOfGuessesInt = inputState[5]
        for sumIdx in range(1,17):
            if sumIdx in sumOfGuessesInt:
                sumOfGuesses.append(1)
            else:
                sumOfGuesses.append(0)

        state = [
            cards,
            cardsOnTable,
            cardsPlayed,
            cardGuess,
            cardsTaken,
            sumOfGuesses,
        ]

        state = list(itertools.chain(*state))
        
        return np.array(state,dtype=int)

    def remember(self, state,next_state,cardToPlay,reward,done,legalMoves):
        self.memory.append((state,next_state,cardToPlay,reward,done,legalMoves)) # popleft if max memory reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        
        states,next_states,cardToPlays,rewards,dones,legalMoves = zip(*mini_sample)
        self.trainer.train_step(states,next_states,cardToPlays,rewards,dones,legalMoves)

    def train_short_memory(self,state,next_state,cardToPlay,reward,done,legalMoves):
        self.trainer.train_step(state,next_state,cardToPlay,reward,done,legalMoves)

    def get_action(self,state,legalMoves):
        '''
        Output: Card to play (52) 
        '''
        self.epsilon = np.exp(-self.n_games*0.01)

        cardToPlay = [0] * 52
        if np.random.rand(1) < self.epsilon and self.training:
             card = random.choice(legalMoves) - 1
             cardToPlay[card] = 1
             return cardToPlay
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0)
        
        sorted_prediction = sorted(prediction, reverse=True)

        for i, value in enumerate(sorted_prediction): # loop through and find best legal move
            cards = ((prediction == value).nonzero(as_tuple=True)[0])
            for card in cards:
                if card + 1 in legalMoves:
                    cardToPlay[card] = 1
                    return cardToPlay


    def trainMove(self,inputState,legalMoves,done):
        
        # get current state
        next_state_input = inputState.copy()
        
        self.state_curr = self.get_state(inputState)

        # get move
        self.cardToPlayTrain = self.get_action(self.state_curr,legalMoves)
        self.cardToPlay = self.cardToPlayTrain.index(1) + 1

        # next_state after action
        next_state_input[0].pop(next_state_input[0].index(self.cardToPlay))
        next_state_input[1].append(self.cardToPlay)
        next_state_input[2].append(self.cardToPlay)

        self.next_state = self.get_state(next_state_input)

        
        self.round.append((self.state_curr,self.next_state,[self.cardToPlay - 1],[done],legalMoves))
        return self.cardToPlay

    def trainSave(self,reward,name,nr_of_cards,game_over):
        for roundNr in range(nr_of_cards):
            state,next_state,cardToPlay,done,legalMoves = zip(self.round[roundNr])

            self.train_short_memory(state,next_state,cardToPlay,[reward],done,legalMoves)

            self.remember(state,next_state,cardToPlay,[reward],done,legalMoves)

        if reward != 0:
            self.win += 1
            self.round_score.append(1)
        else:
            self.round_score.append(0)
        
        self.score += reward
        

        # delete all content from saved rounds
        self.round.clear()

        if game_over:
            self.train_long_memory()
            self.n_games += 1

            if self.score > self.record:
                self.record = self.score
                self.model.save(f'Player_{name}.pth')
            
            self.round_score_tot = list(map(add, self.round_score_tot, self.round_score))
            self.round_score_average = [x/self.n_games for x in self.round_score_tot]

            print('Game:',self.n_games,'Score',self.score,'Record',self.record)

            self.plot_score.append(self.win/17)
            self.total_score += self.win/17
            mean_score = self.total_score/ self.n_games
            self.plot_mean_scores.append(mean_score)
            self.plotter.plot(self.plot_score,self.plot_mean_scores,int(name),f'Player_{name}')
            self.plotter.plotInidivualRounds(self.round_score_average,int(name),f'Player_{name}')
            self.round_score.clear()
            self.score = 0
            self.win = 0
           