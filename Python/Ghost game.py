"""
Assignment 6 Problem 2: Ghost game
October 10, 2024
Kiranmayie Bethi
"""

import random
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
    with open(wordlist_file) as f:                        # call file, read file to list
        wordlist = [word.lower() for word in f.read().splitlines()]
    print("  ", len(wordlist), "words loaded.") 
    return wordlist


def into_dictionary(sequence):
    """
    Returns a dictionary where the keys are elements of the sequence
    and the values are integer counts, for the number of times that
    an element is repeated in the sequence.
    sequence: string or list
    return: dictionary
    """
    # freqs: dictionary (element_type -> int)
    freq = {}
    for x in sequence:
        freq[x] = freq.get(x, 0) + 1
    return freq


# end of helper functions
# -----------------------------------

# Load the word dictionary by assignment the file name to 
# the wordlist variable 
word_list = import_wordlist()

def is_valid_fragment(fragment, word_list):
    """Check if any word in the word list starts with the given fragment."""
    for word in word_list:
        if word.startswith(fragment):
            return True
    return False

def is_complete_word(word, word_list):
    """Check if the word is in the word list and is longer than three letters."""
    return len(word) > 3 and word in word_list

def ghost_game():
    """Main function to run the Ghost game."""
    if not word_list:
        print("The game cannot start without a valid word list.")
        return

    print("Welcome to Ghost! Two players take turns adding letters to form a word fragment.")
    fragment = ""
    current_player = 1

    while True:
        print(f"\nCurrent fragment: {fragment}")
        print(f"Player {current_player}'s turn.")
        
        # Get the next letter from the current player
        letter = input("Enter your letter: ").lower()

        # Validate the letter input
        if len(letter) != 1 or not letter.isalpha():
            print("Invalid input. Please enter a single letter.")
            continue
        
        # Add the letter to the fragment
        fragment += letter

        # Check if the fragment completes a valid word longer than three letters
        if is_complete_word(fragment, word_list):
            print(f"Player {current_player} loses! '{fragment}' is a complete word.")
            break

        # Check if the fragment is a valid start for any word
        if not is_valid_fragment(fragment, word_list):
            print(f"Player {current_player} loses! No words can be formed with the fragment '{fragment}'.")
            break

        # Switch to the other player
        current_player = 2 if current_player == 1 else 1

    print("Game over!")

if __name__ == "__main__":
    ghost_game()