# Example file showing a circle moving on screen
import pygame
from pygame.locals import*
import numpy as np
import random
import os
import time

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

    def __init__(self,playerName):
        self.name = playerName
        self.cards = []
        self.cardGuess = 0
        self.points = 0
        self.positionIdx = 0
        self.cardsTaken = 0

    def displayPlayersCards(self,playCard = False):
        '''
        '''
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
        for card in range(nr_of_cards):
            DealtCard = CardDeck.pop(0)
            self.cards.append(DealtCard)

        self.cards.sort()
        print('Here are your cards:')
        print(self.cards)
        self.displayPlayersCards()

    def guessCardsToTake(self,SumOfGuesses,nr_of_players):
        '''
        '''
        guessString = f"{self.name} guess how many cards you want to take (0-{len(self.cards)-1})"
        text = font.render(guessString, True, black)
        screen.blit(text, (SCREEN_WIDTH/2-200, SCREEN_HEIGHT/2))
        pygame.display.update()
        print('Guess how many cards:')
        
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            if event.type == KEYDOWN and event.key >= 48 and event.key <= 57:
                cardGuess = event.key-48

                if self.positionIdx == nr_of_players and SumOfGuesses+cardGuess == len(self.cards):
                    print("Last to guess, sum of guesses cant equal number of cards")
                else:
                    self.cardGuess = cardGuess
                    return cardGuess
        
    def make_a_move(self):
        '''

        '''
        self.displayPlayersCards(True)

        print('Player',self.name, 'play a card, your cards are: ', self.cards)
        pygame.event.clear()
        while True:
            event = pygame.event.wait()
            if event.type == KEYDOWN:
                cardIdxChosen = event.key-48
                if cardIdxChosen <= len(self.cards)-1 and cardIdxChosen >= 0:
                    cardChosen = self.cards.pop(cardIdxChosen)
                    return cardChosen 
                else:
                    print('Not a valid card, pick a card you have')

class Game():

    def __init__(self,nr_of_players,nr_of_cards):
        self.nr_of_players = nr_of_players
        self.nr_of_cards = nr_of_cards
        self.game_done = False
        self.sumOfGuesses = 0

        self.players = []
        for player in range(self.nr_of_players):
            player_name = f'player_{player+1}'
            player_x = Player(player_name)
            self.players.append(player_x)

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
        self.SumOfGuesses = 0
        for player in self.players:
            self.displayMainScreen(self.players)

            player.positionIdx = player_idx

            player.assignCards(CardDeck,self.nr_of_cards) # Deal card to player

            # Player guesses nr of cards, with knowledge of how many cards have been guessed before him
            playerGuess = player.guessCardsToTake(self.sumOfGuesses,self.nr_of_players) 

            self.SumOfGuesses += playerGuess # Update sum of guesses

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

        for turn in range(self.nr_of_cards):
            cardsOnTable = {}
            for player in self.players:
                self.displayMainScreen(playersDisplay,cardsOnTable)
                cardsOnTable[player.name] = player.make_a_move() # Each player makes a move

            winningPlayer = self.evaluateWinnerOfCard(cardsOnTable) # Assign winner of the card
            playerCardWinner = next((player for player in self.players if player.name == winningPlayer), None)
            playerCardWinner.cardsTaken += 1
            print('Player',playerCardWinner.name,'won the card')

            self.reorderPlayersBasedOnWinner(playerCardWinner)

            self.displayMainScreen(playersDisplay,cardsOnTable)
            
            cardWonString = f"{playerCardWinner.name} won the card. Press any button to continue."
            Text = font.render(cardWonString, True, black)
            screen.blit(Text, (SCREEN_WIDTH/2-200, SCREEN_HEIGHT/2-200))
            pygame.display.update()

            while True:
                event = pygame.event.wait()
                if event.type == KEYDOWN:
                    break

    def assignPoints(self):
        for player in self.players:
            if player.cardGuess == player.cardsTaken:
                if player.cardGuess == 0:
                    player.points += 5
                else:
                    player.points += player.cardGuess + 10

    def resetPlayersGuess(self):
        for player in self.players:
            player.cardsTaken = 0
            player.cardGuess = 0

    def main(self):
        
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        self.displayMainScreen(self.players)

        for cardRound in range(self.nr_of_cards-1):
            
            CardDeck = random.sample(range(1,53), 52) # Shuffle cards

            # Deal cards to every player
            self.dealCards(CardDeck)
            
            self.displayMainScreen(self.players)

            # Play out the round, loop through nr of cards
            self.playRound()

            self.displayMainScreen(self.players)

            # Assign points for the round
            self.assignPoints()

            # Reset guesses and cards taken, only for visuals
            self.resetPlayersGuess()

            self.players.append(self.players.pop(0)) # Move first player to last
            self.nr_of_cards -= 1

        self.game_done = True

if __name__ == '__main__':
    nr_of_cards = 10
    nr_of_players = 3

    game = Game(nr_of_players,nr_of_cards)

    game.main()