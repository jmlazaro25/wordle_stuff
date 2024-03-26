from random import choice
from collections import Counter
from collections import defaultdict

from numpy import ndarray
from numpy import zeros

if __name__ == '__main__':
    from argparse import ArgumentParser


ALPH: str = 'ABCDEFGHJIKLMNOPQRSTUVWXYZ'
ALPH_IDS: dict[str, int] = {char: i + 1 for i, char in enumerate(ALPH)}


class Wordle():
    """
    State values:
        Rows correspond to attempts
        Even indexed columns correspond to letter spaces.
            E.g. self.state[0][0] = 1 means the first guess had an "a" in as the
            first letter.
        Odd indexed columns correspond to assesments/hints.
            E.g. self.state[0][1] = 1 means that there is an "a" in the target
            word, but it is in another space.
        The codes for these columns is as follows:
            -2: Character and hint space not used for short target word
            -1: Character not in target word.
             0: Initial, "empty", state.
             1: Character used in target word, but in a different space.
             2: Character placed correctly.
    """
    max_attempts = 6
    max_chars = 15
    def __init__(self, words: list[str], target_len: int = None) -> None:
        """ Choose target and initialize state """
        if target_len:
            self.target_len = target_len
            words = tuple(w for w in words if len(w) == target_len)
            self.target = choice(words)
        else:
            words = (w for w in words if len(w) > Wordle.max_chars)
            self.target = choice(words)
            self.target_len = len(self.target)

        self.target_counts = Counter(self.target)

        self.attempts_made = 0
        self.state = zeros((Wordle.max_attempts, Wordle.max_chars * 2))
        self.state[:, self.target_len * 2:] = -2

    def guess(self, word: str) -> ndarray:
        """
        Add (and return) guess assesment to state
        Clips word if longer longer than self.target
        """
        word = word.upper()[:self.target_len]
        char_counts = defaultdict(lambda: 0)
        for i, char in enumerate(word):
            char_space = 2 * i
            hint_space = char_space + 1
            char_counts[char] += 1
            self.state[self.attempts_made, char_space] = ALPH_IDS[char]

            if char == self.target[i]:
                self.state[self.attempts_made, hint_space] = 2
            elif char in self.target:
                if char_counts[char] <= self.target_counts[char]:
                    self.state[self.attempts_made, hint_space] = 1
                else:
                    self.state[self.attempts_made, hint_space] = -1
            else:
                self.state[self.attempts_made, hint_space] = -1

        self.attempts_made += 1
        return self.state

    def check_state(self) -> int:
        """ Get state code: -1 = Lost, 0 = ongoing, 1 = Won """
        recent_attempt_success = all(
            self.state[self.attempts_made - 1, hint] == 2
            for hint in (2 * i + 1 for i in range(self.target_len))
        )
        if recent_attempt_success:
            return 1
        elif self.attempts_made < Wordle.max_attempts:
            return 0
        else:
            return -1


def main() -> None:
    """ Play Wordle on command line - FOR TESTING - NOT USER FRIENDLY """
    parser = ArgumentParser()
    parser.add_argument('-l', '--wlen', type=int, default=None)
    parser.add_argument('--vocab_file', default='corncob_caps.txt')
    args = parser.parse_args()

    with open(args.vocab_file) as f:
        all_words = tuple(f.read().split('\n'))

    game = Wordle(all_words, target_len=args.wlen)
    while game.check_state() == 0:
        guess = input(
            f'Guess a {game.target_len}-letter word '
            + f'(attempt {game.attempts_made + 1}): '
        )
        game.guess(guess)
        print('Current state:')
        print(game.state[:game.attempts_made, :2 * game.target_len])

    if game.check_state() == 1:
        print('You Won!')
    else:
        print(f'You lost :( The target word was {game.target}.')


if __name__ == '__main__':
    main()
