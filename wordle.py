from random import choice
from collections import Counter
from collections import defaultdict
from typing import Union

from numpy import zeros
from numpy import ndarray
from numpy import int64
from numpy import float64

if __name__ == '__main__':
    from argparse import ArgumentParser


ALPH: str = 'ABCDEFGHJIKLMNOPQRSTUVWXYZ'
ALPH_IDS: dict[str, int] = {char: i + 1 for i, char in enumerate(ALPH)}
ALPH_LEN: int = len(ALPH)


class Wordle():
    """
    State values: (char_hint_code, attempt, character_space)
        Character codes are given by ALPH_IDS
        Hint codes are follows:
            0: Initial, "empty", state.
            1: Character not in target word.
            2: Character used in target word, but in a different space.
            3: Character placed correctly.
        E.g. self.state[:, 0, 0]  = [1, 3] means the first guess had an "a" in
             as the first letter and it was correct
    State (summary) codes:
        -1 = Lost
         0 = Ongoing
         1 = Won
    """
    char_channel = 0
    hint_channel = 1
    n_channels = 2
    max_attempts = 6
    inital_empty = 0
    not_used = 1
    wrong_place = 2
    correct = 3
    n_hint_states = 4
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
            self.target = choice(tuple(words))
            self.target_len = len(self.target)

        self.target_counts = Counter(self.target)

        self.attempts_made = 0
        self.state = zeros(
            (Wordle.n_channels, Wordle.max_attempts, self.target_len),
            dtype=int
        )

    def guess(
            self,
            word: str,
            possible_words: Union[tuple[str], set[str]] = None
        ) -> ndarray[int64]:
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
        # twice in the guess, only used once in the target, and the later use is
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
                self.state[*space_tuple] = Wordle.wrong_place
            else:
                self.state[*space_tuple] = Wordle.not_used

        self.attempts_made += 1
        return self.state

    def one_hot_guess(
            self,
            word: str,
            possible_words: Union[tuple[str], set[str]] = None
        ) -> tuple[ndarray[float64]]:
        return Wordle.one_hot_state(
            self.guess(word, possible_words),
            self.target_len,
            self.attempts_made
        )

    @staticmethod
    def one_hot_state(
            state: ndarray[int64],
            target_len: int,
            attempts_made: int
        ) -> tuple[ndarray[float64]]:
        one_hot_chars = zeros((ALPH_LEN + 1, Wordle.max_attempts, target_len))
        one_hot_hints = zeros(
            (Wordle.n_hint_states, Wordle.max_attempts, target_len)
        )
        for attempt_i in range(attempts_made):
            for pos_i in range(target_len):
                space_tuple = (attempt_i, pos_i)
                char, hint = state[:, *space_tuple]
                one_hot_chars[char, *space_tuple] = 1.
                one_hot_hints[hint, *space_tuple] = 1.
        return (one_hot_chars, one_hot_hints)

    def check_state(self) -> int:
        recent_attempt_success = all(
            self.state[
                Wordle.hint_channel,
                self.attempts_made - 1
            ] == Wordle.correct
        )
        if recent_attempt_success:
            return Wordle.won
        elif self.attempts_made < Wordle.max_attempts:
            return Wordle.ongoing
        else:
            return Wordle.lost


def load_vocab(
        vocab_file: str ='corncob_caps.txt',
        data_struct: type = tuple
    ) -> Union[tuple[str], set[str]]:
    with open(vocab_file) as f:
        return data_struct(f.read().upper().split('\n'))


def main() -> None:
    """ Play Wordle on command line - FOR TESTING - NOT USER FRIENDLY """
    parser = ArgumentParser()
    parser.add_argument('-l', '--wlen', type=int, default=None)
    parser.add_argument('-w', '--test_word')
    parser.add_argument('--vocab_file', default='corncob_caps.txt')
    parser.add_argument('-r', '--restrict_guesses', action='store_true')
    parser.add_argument('-o', '--one_hot', action='store_true')
    args = parser.parse_args()

    all_words_data_struct = set if args.restrict_guesses else tuple

    if args.test_word:
        all_words = all_words_data_struct([args.test_word.upper()])
    else:
        with open(args.vocab_file) as f:
            all_words = load_vocab(
                args.vocab_file,
                data_struct=all_words_data_struct
            )

    possible_words = all_words if args.restrict_guesses else None

    game = Wordle(all_words, target_len=args.wlen)
    while game.check_state() == Wordle.ongoing:
        guess = input(
            f'Guess a {game.target_len}-letter word '
            + f'(attempt {game.attempts_made + 1}): '
        )

        print('Current state:')
        if not args.one_hot:
            game.guess(guess, possible_words)
            print(game.state[:, :game.attempts_made])
        else:
            chars, hints = game.one_hot_guess(guess, possible_words)
            print('Characters:')
            print(chars[:, :game.attempts_made])
            print('Hints:')
            print(hints[:, :game.attempts_made])

    if game.check_state() == Wordle.won:
        print('You Won!')
    else:
        print(f'You lost :( The target word was {game.target}.')


if __name__ == '__main__':
    main()
