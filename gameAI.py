# Example file showing a circle moving on screen
import pygame
from pygame.locals import*
import numpy as np
import random
import os
import time
from guessAgent import GuessAgent
from playAgent import PlayAgent
import itertools

# pygame setup
pygame.init()
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
running = True
dt = 0
font = pygame.font.SysFont(None, 24)

def loadCardImages():
    '''
    '''
    path_to_directory = '/Users/emilbostrom/Documents/python/Plump_AI/PlayingCards'
    image_dict = {}
    for filename in os.listdir(path_to_directory):
        if filename.endswith('.png'):
            path = os.path.join(path_to_directory, filename)
            key = filename[:-4]
            img = pygame.image.load(path).convert()
            scaledImg = pygame.transform.scale(img, (128, 186))
            image_dict[key] = scaledImg
    
    return image_dict


cardImagesDict = loadCardImages()

green = (53, 101, 77)
black = (0,0,0)

class Player():
    '''
    Player class. Player has a list of cards, a card guess, nr of cards taken during the round,
    a number of points from winning rounds, a position at the table for the specific round.

    Has functions for being dealt cards, guessing cards to take and making a move.
    '''

    def __init__(self,playerName,AIPlayer):
        self.name = playerName
        self.AI = AIPlayer
        self.cards = []
        self.cardGuess = 0
        self.points = 0
        self.positionIdx = 0
        self.cardsTaken = 0
        self.playAgent = PlayAgent(self.name)
        self.guessAgent = GuessAgent(self.name)

    def displayPlayersCards(self,playCard = False):
        '''
        '''
        if self.AI:
            return

        card_x = 0
        card_y = 0
        
        for card in self.cards:
            screen.blit(cardImagesDict[str(card)],(card_x,card_y))
            card_x += 128
        
        if playCard:
            playString = f"{self.name} play a card (0-{len(self.cards)-1})"
            text = font.render(playString, True, black)
            screen.blit(text, (SCREEN_WIDTH/2-100, SCREEN_HEIGHT/2-100))
        
        pygame.display.update()

    def assignCards(self,CardDeck,nr_of_cards):
        '''
        '''
        self.cards = []
        for card in range(nr_of_cards):
            DealtCard = CardDeck.pop(0)
            self.cards.append(DealtCard)

        self.cards.sort()
        self.displayPlayersCards()

    def guessCardsToTakeAI(self,SumOfGuesses,nr_of_players,training):
        '''
        '''

        legalMoves = [0,1,2,3,4,5,6,7,8,9,10]
        if self.positionIdx == nr_of_players:
            forbiddenGuess = len(self.cards) - SumOfGuesses
            if forbiddenGuess in legalMoves:
                legalMoves.remove(forbiddenGuess)

        if self.AI:
            inputState = [self.cards, [SumOfGuesses],[self.positionIdx]]
            inputState = list(itertools.chain(*inputState))

            #if training:
            #    cardGuess = self.guessAgent.trainMove(inputState,legalMoves)
            #else:
            state_curr,done = self.guessAgent.get_state(inputState)
            cardGuess = self.guessAgent.get_action(state_curr,legalMoves)
                
            self.cardGuess = cardGuess # Turn 0:s and 1:s to actual int
            return self.cardGuess
        else:
            while True:
                guessString = f"{self.name} guess how many cards you want to take (0-{len(self.cards)-1})"
                text = font.render(guessString, True, black)
                screen.blit(text, (SCREEN_WIDTH/2-200, SCREEN_HEIGHT/2))
                pygame.display.update()
                
                pygame.event.clear()
                event = pygame.event.wait()
                if event.type == KEYDOWN and event.key >= 48 and event.key <= 57:
                    cardGuess = event.key-48
                    if cardGuess not in legalMoves:
                        string = f"Last to guess, sum of guesses cant equal number of cards"
                        text = font.render(string, True, black)
                        screen.blit(text, (SCREEN_WIDTH/2-200, SCREEN_HEIGHT/2-100))
                        pygame.display.update()
                    else:
                        self.cardGuess = cardGuess
                        return self.cardGuess

    def getLegalMoves(self,cardsOnTable):
        legalMoves = []
        if not cardsOnTable:
            return self.cards
        else:
            for card in self.cards:
                if np.floor(np.divide(cardsOnTable[0]-1,13)) == np.floor(np.divide(card-1,13)):
                    legalMoves.append(card)
            if not legalMoves: # if the suit played is not in the hand all cards are legal
                return self.cards
            else:
                return legalMoves

    def make_a_move(self,cardsOnTable,cardsPlayed,game_done,sumOfGuesses,training):
        '''

        '''
        
        if self.AI:
            legalMoves = self.getLegalMoves(cardsOnTable)
            inputState = [legalMoves, cardsOnTable, cardsPlayed, [self.cardGuess], [self.cardsTaken],[sumOfGuesses]]

            if training:
                cardToPlay = int(self.playAgent.trainMove(inputState,legalMoves,game_done))
            else:
                state_curr = self.playAgent.get_state(inputState)
                cardToPlay = self.playAgent.get_action(state_curr,legalMoves)
                cardToPlay = cardToPlay.index(1) + 1
                self.cards.remove(cardToPlay)


            return cardToPlay
        else:    
            self.displayPlayersCards(True)
            pygame.event.clear()
            while True:
                event = pygame.event.wait()
                if event.type == KEYDOWN:
                    cardIdxChosen = event.key-48
                    if cardIdxChosen <= len(self.cards)-1 and cardIdxChosen >= 0:
                        cardToPlay = self.cards.pop(cardIdxChosen)
                        return cardToPlay 
                    else:
                        print('Not a valid card, pick a card you have')

class Game():

    def __init__(self,initPlayers):
        self.nr_of_players = len(initPlayers)
        self.nr_of_cards = 10
        self.game_done = False
        self.sumOfGuesses = 0
        self.cardsPlayed = []
        self.rounds = [9,8,7,6,5,4,3,2,2,3,4,5,6,7,8,9,10]
        self.training = False

        self.players = []
        for playerAI in initPlayers:
            player_name = f'{len(self.players)+1}'
            player_x = Player(player_name,playerAI)
            self.players.append(player_x)
        
        self.playersDealOrder = self.players

    def displayMainScreen(self,playersDisplay, cardsOnTable = {}):
        
        # fill the screen with a color to wipe away anything from last frame
        screen.fill(green)

        playerIdx = 0
        for player in self.players:
            guessString = f"{player.name} guess: {player.cardGuess}"
            guessText = font.render(guessString, True, black)
            screen.blit(guessText, (0, SCREEN_HEIGHT - 30 - playerIdx*20))

            cardsTakenString = f"{player.name} cards taken: {player.cardsTaken}"
            cardsTakenText = font.render(cardsTakenString, True, black)
            screen.blit(cardsTakenText, (200, SCREEN_HEIGHT - 30 - playerIdx*20))

            pointsString = f"{player.name} points: {player.points}"
            pointsText = font.render(pointsString, True, black)
            screen.blit(pointsText, (400, SCREEN_HEIGHT - 30 - playerIdx*20))

            card_x = 128*3
            card_y = 300
            for card in cardsOnTable.values():
                screen.blit(cardImagesDict[str(card)],(card_x,card_y))
                card_x += 128

            playerIdx += 1
        
        pygame.display.update()   

    def evaluateWinnerOfCard(self, cardsOnTable):
        '''
        cardsOnTable: Dictionary with all players and their respective card they 
        put on the table for the round
        '''
        firstPlayer = True

        for player in cardsOnTable:
            if firstPlayer:
                leading_card = cardsOnTable[player]
                leading_player = player
                firstPlayer = False
            else:
                sameSuit = np.floor(np.divide(cardsOnTable[player]-1,13)) == np.floor(np.divide(leading_card-1,13))
                higherValue = cardsOnTable[player] > leading_card
                if sameSuit and higherValue:
                    leading_card = cardsOnTable[player]
                    leading_player = player
        
        return leading_player

    def reorderPlayersBasedOnWinner(self,winningPlayer):

        # Find the index of the winning player
        winningIndex = self.players.index(winningPlayer)

        # Reorder the list
        self.players = [self.players[winningIndex]] + self.players[winningIndex+1:] + self.players[:winningIndex]

    def dealCards(self,CardDeck):
        
        player_idx = 1
        self.sumOfGuesses = 0
        for player in self.players:
            self.displayMainScreen(self.players)

            player.positionIdx = player_idx

            player.assignCards(CardDeck,self.nr_of_cards) # Deal card to player

            # Player guesses nr of cards, with knowledge of how many cards have been guessed before him
            playerGuess = player.guessCardsToTakeAI(self.sumOfGuesses,self.nr_of_players,self.training) 

            f = open("guessTraining.txt", "a")
            guessString = f"{player.cards} {self.sumOfGuesses} {player_idx}\n"
            f.write(guessString)
            f.close()

            f = open("guessResults.txt", "a")
            resultsString = f"{playerGuess}\n"
            f.write(resultsString)
            f.close()

            self.sumOfGuesses += playerGuess # Update sum of guesses
            player_idx += 1

    def playRound(self):
        '''
        '''
        
        # Keep order of players for displaying
        playersDisplay = self.players

        # Hihgest initial guess starts
        playerHighestGuess = self.players[0]
        for player in self.players[1:]:
            if player.cardGuess > playerHighestGuess.cardGuess:
                playerHighestGuess = player

        self.reorderPlayersBasedOnWinner(playerHighestGuess)

        cardsPlayed = []
        for turn in range(self.nr_of_cards):
            cardsOnTable = {}
            for player in self.players:
                self.displayMainScreen(playersDisplay,cardsOnTable)
                cardsOnTable[player.name] = player.make_a_move(list(cardsOnTable.values()),cardsPlayed,self.game_done,self.sumOfGuesses,self.training) # Each player makes a move
                cardsPlayed.append(cardsOnTable[player.name])

            winningPlayer = self.evaluateWinnerOfCard(cardsOnTable) # Assign winner of the card
            playerCardWinner = next((player for player in self.players if player.name == winningPlayer), None)
            playerCardWinner.cardsTaken += 1

            self.reorderPlayersBasedOnWinner(playerCardWinner)

            self.displayMainScreen(playersDisplay,cardsOnTable)
            
            cardWonString = f"{playerCardWinner.name} won the card. Press any button to continue."
            Text = font.render(cardWonString, True, black)
            screen.blit(Text, (SCREEN_WIDTH/2-200, SCREEN_HEIGHT/2-200))

            if not self.training:
                while True:
                    pygame.display.update()
                    event = pygame.event.wait()
                    if event.type == KEYDOWN:
                        break

    def assignPoints(self):
        for player in self.playersDealOrder:
            if player.cardGuess == player.cardsTaken:
                if player.cardGuess == 0:
                    player_reward = 5
                    player.points += player_reward
                    resultsString = f"5\n"
                else:
                    player_reward = 10 + player.cardGuess
                    player.points += player_reward
                    resultsString = f"{player.cardGuess + 10}\n"
            else:
                player_reward = 0
                resultsString = f"0\n"
            
            if self.training:
                player.playAgent.trainSave(player_reward,player.name,self.nr_of_cards,self.game_done)
                #player.guessAgent.trainSave(player_reward,player.name,self.game_done)

    def resetPlayersGuess(self):
        for player in self.players:
            player.cardsTaken = 0
            player.cardGuess = 0

    def resetGame(self):
        for player in self.players:
            player.cards = []
            player.cardGuess = 0
            player.points = 0
            player.positionIdx = 0
            player.cardsTaken = 0
        self.nr_of_cards = 10
        self.game_done = False
        self.sumOfGuesses = 0
        self.cardsPlayed = []
        self.main()

    def main(self):
        
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        self.displayMainScreen(self.players)

        for cardRound in self.rounds:
            
            if cardRound == 10:
                self.game_done = True
            
            self.nr_of_cards = cardRound
            
            CardDeck = random.sample(range(1,53), 52) # Shuffle cards

            # Deal cards to every player
            self.dealCards(CardDeck)
            
            self.displayMainScreen(self.players)

            # Play out the round, loop through nr of cards
            #self.playRound()

            self.displayMainScreen(self.players)

            # Assign points for the round
            #self.assignPoints()

            # Reset guesses and cards taken, only for visuals
            self.resetPlayersGuess()

            self.playersDealOrder.append(self.playersDealOrder.pop(0)) # Move first player to last
            self.players = self.playersDealOrder
        
        self.resetGame()

if __name__ == '__main__':
    players = [0, 0, 0, 0, 0] # 0 is human 1 is AI

    game = Game(players)

    game.main()