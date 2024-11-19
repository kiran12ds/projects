"""
Assignment 6 Problem 1: Scrabble game
October 10, 2024
Kiranmayie Bethi
"""

import random

vowels = 'aeiou'
not_vowels = 'bcdfghjklmnpqrstvwxyz'
letters_per_hand = 7

points_by_letter = {
    'a': 1, 'b': 3, 'c': 3, 'd': 2, 'e': 1, 'f': 4, 'g': 2, 'h': 4, 'i': 1, 'j': 8, 'k': 5, 'l': 1, 'm': 3, 'n': 1, 'o': 1, 'p': 3, 'q': 10, 'r': 1, 's': 1, 't': 1, 'u': 1, 'v': 4, 'w': 4, 'x': 8, 'y': 4, 'z': 10
}


# -----------------------------------
# Helper functions
# (you don't need to understand this code)

wordlist_file = "words.txt"

def import_wordlist():
    """
    Imports a list of words from external file
    Returns a list of valid words for the game
    Words are all in lowercase letters
    """
    print("Loading word list from file...")
    with open(wordlist_file) as f:                 # call file, read file to list
        wordlist = [word.lower() for word in f.read().splitlines()]
    print("  ", len(wordlist), "words loaded.") 
    return wordlist

def into_dictionary(sequence):
    """
    Returns a dictionary where the keys are elements of the sequence
    and the values are integer counts, for the number of times that
    an element is repeated in the sequence.
    sequence: str or list
    return: dictionary
    """
    # freqs: dictionary (values type: int)
    freq = {}
    for letter in sequence:
        freq[letter] = freq.get(letter, 0) + 1
    return freq


# end of helper functions
# -----------------------------------

# -----------------------------------
# Problem #1: Scoring a word

def calc_word_score(word, qty):
    """
    Returns the word score after word is validated.
    The score for a word is the sum of the points for letters
    in the word, plus 50 points if all the letters in hand are used
    word: string (lowercase letters)
    returns: int >= 0
    """
    # Total score calculation
    score = sum(points_by_letter.get(letter, 0) for letter in word)

    # Bonus for using all letters in the hand
    if len(word) == qty:
        score += 50

    return score


# make sure you understand how this function works and what it does;
#    it will help with the work you have to do

def show_hand(hand):
    """
    Prints the letters in the player's hand
    For example:
       show_hand({'a':1, 'f':2, 'n':2, 'e':2})
    Prints something like:
       a f f n n e e
    The order of the letters is unimportant
    hand: dictionary (string -> int)
    """
    for letter in hand.keys():
        for j in range(hand[letter]):   # calling the letters one by one from the dictionary
            print(letter, end = " "),   # print all on the same line
    print(" ")                          # print an empty line


# make sure you understand how this function works and what it does;
#    it will help with the work you have to do
def dealing_hands(qty):
    """
    Returns a random hand with qty lowercase letters for hand
    a third of letters are vowels
    the letters and letter frequencies are stored in a dictionary
    key = letter; value = frequency
    qty: int >= 0
    returns: dictionary (string -> int)
    """
    hand={}
    num_vowels = qty // 3         # changed from divide to floor
    # collect the vowels
    for i in range(num_vowels):
        letter = vowels[random.randrange(0, len(vowels))]
        hand[letter] = hand.get(letter, 0) + 1
    # collect the consonants
    for i in range(num_vowels, qty):    
        letter = not_vowels[random.randrange(0, len(not_vowels))]
        hand[letter] = hand.get(letter, 0) + 1
        
    return hand

# -----------------------------------
# Problem #2: Update the hand by removing letters

def hand_update(hand, word):
    """
    After word played and validated, 
    removes letters in word from hand
    if hand has 2 a's & an 'a' was used,
    this updates hand to 1 'a'
    word: string
    hand: dictionary (string -> int)    
    returns: dictionary (string -> int)
    """
    """
    Updates the hand by removing the letters in the given word.
    Assumes that all the letters in the word are valid and appear in the hand.

    hand: dictionary (string -> int) representing the current hand
    word: string representing the word to be formed
    returns: updated hand dictionary (string -> int)
    """
    # Create a copy of the hand to avoid modifying the original hand
    updated_hand = hand.copy()

    # Update the hand by removing the letters used in the word
    for letter in word:
        if updated_hand.get(letter, 0) > 0:
            updated_hand[letter] -= 1
            # Remove the letter from the dictionary if its count is zero
            if updated_hand[letter] == 0:
                del updated_hand[letter]

    return updated_hand
# -----------------------------------#
# Problem #3: Test the word validity

def word_is_valid(word, hand, word_list):
    """
    Returns boolean
    if all the letters in the word played are in the hand
    and
    if the word is in the wordlist
    returns true
    if either false, returns false
    word: string
    hand: dictionary (string -> int)
    word_list: list of lowercase words
    """
     # Check if the word is in the word list
    if word not in word_list:
        return False

    # Check if the hand has enough letters for the word
    hand_copy = hand.copy()  # Make a copy to avoid modifying the original hand
    for letter in word:
        if hand_copy.get(letter, 0) <= 0:
            return False
        hand_copy[letter] -= 1

    return True
# -----------------------------------
# Problem #4: Playing a hand

def playing_hands(hand, word_list):
    """
    Allows the user to play the given hand, as follows:
    * hand shown
    * user can play a word from hand
    * invalid words are rejected with a message to player to play a different word
    * if valid word, remove letters from hand
    * if valid word scores word, adds score to total score
    * total score is shown to player after each valid word is scored
    * then the hand is shown, followed by asking user to play another word
    * hand is over when no remaining letters
    * user can stop game by entering a . instead of a word (the period)
    * if game ended, final score is shown
      hand: dictionary (string -> int)
      word_list: list of lowercase strings
    """
    total_score = 0
    current_hand = hand.copy()
    initial_hand_size = sum(hand.values())  # Store the initial hand size

    # Display the initial hand
    print("Current hand:", ' '.join([letter * count for letter, count in current_hand.items()]))
    print("Given the letters in your hand, make a word to earn points or enter '.' to end your game.")

    while True:
        # Ask the player to input a word, or "." to indicate they are finished
        word = input().strip().lower()

        # If the player inputs ".", end the game
        if word == ".":
            break

        # Check if the word is valid
        if word_is_valid(word, current_hand, word_list):
            # Calculate the score for the word
            word_score = calc_word_score(word, initial_hand_size)

            # Apply the bonus if the word uses all the letters in the hand
            if len(word) == sum(current_hand.values()):
                print("You earned an additional 50 points for using all of the letters in one word.")

            # Update the total score
            total_score += word_score
            print(f"The word {word} got you {word_score} points. Your total score is {total_score}.")

            # Update the hand by removing the letters used in the word
            current_hand = hand_update(current_hand, word)

            # Display the updated hand if there are still letters left
            if sum(current_hand.values()) > 0:
                print("Current hand:", ' '.join([letter * count for letter, count in current_hand.items()]))
                print("Given the letters in your hand, make a word to earn points or enter '.' to end your game.")
            else:
                print("No more usable letters in the hand.")
                break
        else:
            # If the word is invalid, inform the player
            print(f"'{word}' is not a valid word. Please choose another word.")

    # Display the final score
    print(f"Game over. Total score: {total_score} points.")

# -----------------------------------
# Problem #5: Playing the game
# Make sure you understand how this code works
# 
def start_game(word_list):
    """
    Allow players an arbitrary number of hands
    ask user to enter 'n', 'r', or 'e' for the following options:
    * 'n': new random hand; when hand is played, user is asked to play 'n' or 'e' again
    * 'r': replay the previous hand
    * 'e': exit the game
    * if anything other than n, r, or e is entered, ask let user know the options again
    """
    ##playing_hands(dealing_hands(letters_per_hand), word_list)   # delete this once you've completed Problem #4
    
    ## uncomment the following block of code once you've completed Problem #4
    hand = dealing_hands(letters_per_hand) # random init  /// this is redundant---- this shouldn't be here
    while True:
       user_prompted = input('Enter n to start a new game, r to replay the last hand, or e to end game: ')
       if user_prompted == 'n':
           hand = dealing_hands(letters_per_hand)
           playing_hands(hand.copy(), word_list)
           print
       elif user_prompted == 'r':
           playing_hands(hand.copy(), word_list)
           print
       elif user_prompted == 'e':
           break
       else:
           print 
           "You did not choose from the options provided."

# Used for entire session; this starts the game
if __name__ == '__main__':
    word_list = import_wordlist()
    start_game(word_list)
