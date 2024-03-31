from random import choice
from collections import Counter
from collections import defaultdict
from typing import Union

from numpy import ndarray
from numpy import zeros

if __name__ == '__main__':
    from argparse import ArgumentParser


ALPH: str = 'ABCDEFGHJIKLMNOPQRSTUVWXYZ'
ALPH_IDS: dict[str, int] = {char: i + 1 for i, char in enumerate(ALPH)}


class Wordle():
    """
    State values: (char_hint_code, attempt, character_space)
        Character codes are given by ALPH_IDS
        Hint codes are follows:
            -2: Character space not used for short target word
            -1: Character not in target word.
             0: Initial, "empty", state.
             1: Character used in target word, but in a different space.
             2: Character placed correctly.
        E.g. self.state[:, 0, 0]  = [1, 2] means the first guess had an "a" in
             as the first letter and it was correct
    State (summary) codes:
        -1 = Lost
         0 = Ongoing
         1 = Won
    """
    max_attempts = 6
    max_chars = 15
    char_channel = 0
    hint_channel = 1
    n_channels = 2
    space_not_used = -2
    char_not_used = -1
    inital_empty = 0
    char_elsewhere = 1
    correct = 2
    lost = -1
    ongoing = 0
    won = 1
    def __init__(
            self,
            words: Union[tuple[str], set[str]],
            target_len: int = None
        ) -> None:
        """ Choose target and initialize state """
        if target_len:
            self.target_len = target_len
            words = tuple(w for w in words if len(w) == target_len)
            self.target = choice(words)
        else:
            words = tuple(w for w in words if len(w) <= Wordle.max_chars)
            self.target = choice(words)
            self.target_len = len(self.target)

        self.target_counts = Counter(self.target)

        self.attempts_made = 0
        self.state = zeros(
            (Wordle.n_channels, Wordle.max_attempts, Wordle.max_chars)
        )
        self.state[
            Wordle.hint_channel,
            :,
            self.target_len:
        ] = Wordle.space_not_used

    def guess(
            self,
            word: str,
            possible_words: Union[tuple[str], set[str]] = None
        ) -> ndarray:
        """ Add guess assesment to state (and return) """
        if len(word) != self.target_len:
            raise ValueError

        word = word.upper()

        if possible_words and word not in possible_words:
            raise ValueError

        char_tuples = list(enumerate(word))
        char_counts = defaultdict(lambda: 0)

        # Count correct chars first
        # Prevents characters being marked as misplaced if the character is used
        # twice in the guess, only once in the target and the later use is
        # correct.
        offset = 0
        for char_space in range(self.target_len):
            char = word[char_space]
            space_tuple = (self.attempts_made, char_space)
            self.state[Wordle.char_channel, *space_tuple] = ALPH_IDS[char]
            if char == self.target[char_space]:
                char_counts[char] += 1
                self.state[Wordle.hint_channel, *space_tuple] = Wordle.correct
                char_tuples.pop(char_space - offset)
                offset += 1

        for char_space, char in char_tuples:
            space_tuple = (Wordle.hint_channel, self.attempts_made, char_space)
            char_counts[char] += 1

            if (
                char in self.target
                and char_counts[char] <= self.target_counts[char]
            ):
                self.state[*space_tuple] = Wordle.char_elsewhere
            else:
                self.state[*space_tuple] = Wordle.char_not_used

        self.attempts_made += 1
        return self.state

    def check_state(self) -> int:
        recent_attempt_success = all(
            self.state[
                Wordle.hint_channel,
                self.attempts_made - 1,
                :self.target_len
            ] == 2
        )
        if recent_attempt_success:
            return Wordle.won
        elif self.attempts_made < Wordle.max_attempts:
            return Wordle.ongoing
        else:
            return Wordle.lost


def main() -> None:
    """ Play Wordle on command line - FOR TESTING - NOT USER FRIENDLY """
    parser = ArgumentParser()
    parser.add_argument('-l', '--wlen', type=int, default=None)
    parser.add_argument('-w', '--test_word')
    parser.add_argument('--vocab_file', default='corncob_caps.txt')
    parser.add_argument('-r', '--restrict_guesses', action='store_true')
    args = parser.parse_args()

    all_words_datastruct = set if args.restrict_guesses else tuple

    if args.test_word:
        all_words = all_words_datastruct([args.test_word.upper()])
    else:
        with open(args.vocab_file) as f:
            all_words = all_words_datastruct(f.read().split('\n'))

    possible_words = all_words if args.restrict_guesses else None

    game = Wordle(all_words, target_len=args.wlen)
    while game.check_state() == 0:
        guess = input(
            f'Guess a {game.target_len}-letter word '
            + f'(attempt {game.attempts_made + 1}): '
        )
        game.guess(guess, possible_words=possible_words)
        print('Current state:')
        print(game.state[:, :game.attempts_made, :game.target_len])

    if game.check_state() == 1:
        print('You Won!')
    else:
        print(f'You lost :( The target word was {game.target}.')


if __name__ == '__main__':
    main()
