import re
from collections import Counter
from collections import defaultdict
from argparse import Namespace
from argparse import ArgumentParser


words_tuple = tuple[str]
input_strs = list[str]
char_pos_tuple = tuple[str, int]
char_posses = tuple[char_pos_tuple]
scored_words = tuple[tuple[str, float]]

# https://en.wikipedia.org/wiki/Letter_frequency
CHAR_FREQS: dict = {
    'ENGLISH_DICT': 'ESIARNTOLCDUGPMHBYFVKWZXJQ',
    'ENGLISH_TEXT': 'ETAOINSHRDLCUMWFGYPBVKXJQZ'
}


def clean_input(args: Namespace) -> Namespace:
    """ Make inputs upper and remove problematic characters. """
    for k in vars(args):
        v = getattr(args, k)
        if isinstance(v, str):
            setattr(args, k, re.sub('[^A-Z_]', '', v.upper()))
        elif (
            isinstance(v, list)
            and v
            and isinstance(v[0], str)
        ):
            setattr(
                args,
                k,
                tuple(
                    re.sub(r'[^A-Z\d]', '', s.upper()) for s in v
                )
            )
    return args

def filter_by_wlen(words: words_tuple, wlen: int) -> words_tuple:
    return tuple(w for w in words if len(w) == wlen)

def parse_char_pos(char_pos_strs: input_strs) -> char_posses:
    return tuple((s[0].upper(), int(s[1]) - 1) for s in char_pos_strs)

def filter_by_correct(words: words_tuple, correct: input_strs) -> words_tuple:
    """ Keep only words with correct chars in correct positions. """
    correct = parse_char_pos(correct)
    words_copy = list(words)
    for word in words:
        for char, pos in correct:
            if word[pos] != char:
                words_copy.remove(word)
                break
    return tuple(words_copy)

def filter_by_wrong_place(
        words: words_tuple,
        wrong_place: input_strs
    ) -> words_tuple:
    """
    Keep only words that have all `wrong_place` characters, but in different
    positions.
    """
    wrong_place = parse_char_pos(wrong_place)
    wrong_place_chars = set(char for char, _ in wrong_place)
    words_copy = list(words)
    for word in words:
        if not all(re.search(char, word) for char in wrong_place_chars):
            words_copy.remove(word)
            continue
        for char, pos in wrong_place:
            if word[pos] == char:
                words_copy.remove(word)
    return tuple(words_copy)

def get_present_counts(
        correct: input_strs,
        wrong_place: input_strs
    ) -> dict[str, int]:
    """
    Get characters present in word and the number of times each appears.
    This is the sum of the times it appears in the `correct` arg and 1 if it
    appears in `wrong_place` as we may know multiple wrong positions, but that
    does not mean it appears as many times. There may be more than 1 appearance
    of the character, but as long as it is not also in `not present`, the
    character will not be considered 'maxed', and thus words having more
    appearances of said character than indicated here will not be removed.
    """
    correct_counts = (
        Counter(char for char, _ in parse_char_pos(correct)) if correct
        else {}
        )
    wrong_place_counts = (
        {char: 1 for char, _ in parse_char_pos(wrong_place)} if wrong_place
        else {}
    )
    present_counts = defaultdict(lambda: 0)
    for counts in (correct_counts, wrong_place_counts):
        for char, count in counts.items():
            present_counts[char] += count
    return present_counts

def get_maxed_chars(
        correct_wp_np: tuple[input_strs, input_strs, str]
    ) -> dict[str, int]:
    """
    Compare present_counts and not_present to determine if any characters are
    'maxed,' meaning they will not appear more times than in `present_counts`.
    """
    correct, wrong_place, not_present = correct_wp_np
    present_counts = get_present_counts(correct, wrong_place)
    return {
        char: cnt for char, cnt in present_counts.items()
        if char in not_present
    }

def remove_not_present(
        words: words_tuple,
        correct_wp_np: tuple[input_strs, input_strs, str]
    ) -> words_tuple:
    """
    Remove words that contain characters that are not present. Note, characters
    in `not_present` and `maxed_chars` (obtained through the combination of
    `correct`, `wrong_place`, and `not_present) are removed from `not_present`
    as they must be handled seperately.
    """
    maxed_chars = get_maxed_chars(correct_wp_np)
    not_present = ''.join(c for c in correct_wp_np[2] if c not in maxed_chars)
    if not not_present:
        return words
    pat = f'[{not_present}]'
    return tuple(w for w in words if not re.search(pat, w))

def remove_maxed(
        words: words_tuple,
        correct_wp_np: tuple[input_strs, input_strs, str]
    ) -> words_tuple:
    """
    Remove words with more appearances of a character than the count allowed by
    `get_maxed_chars(correct_wp_np)`.
    """
    maxed_chars = get_maxed_chars(correct_wp_np)
    words_copy = list(words)
    for word in words:
        for char, cnt in maxed_chars.items():
            if len(re.findall(char.upper(), word)) > cnt:
                words_copy.remove(word)
    return tuple(words_copy)

def sort_words(
        words: words_tuple,
        freq_and_weight: tuple[str, float]
    ) -> scored_words:
    """
    Sort words by a score made up of the number of unique characters and the
    weighted frequency of the characters.
    """
    char_freq, freq_weight = freq_and_weight
    char_freq = CHAR_FREQS[char_freq] if char_freq in CHAR_FREQS else char_freq
    char_rank = {c: i for i, c in enumerate(reversed(char_freq))}
    word_ranks = (
        (w, len(set(w)) + sum(char_rank[char] * freq_weight for char in w))
        for w in words
    )
    return sorted(word_ranks, key=lambda tup: tup[1], reverse=True)


def main() -> scored_words:
    """
    1. Parse arguments
        1.1. If a character is used in `args.correct` or `args.wrong_place` and
             included in `not_present`, this will be understood to mean that
             there are no more appearances of the character than
             <number of appearances in `correct`>
             + <1 if char in `wrong_place`>.
        1.2. char_frequency can be a key in CHAR_FREQS or a custom frequency.
    2. Open the vocan file, upper, and split words at new lines.
    3. Clean parsed args.
    4. Run steps (currently):
        4.1. Filter for correct word length.
        4.2. Remove words with characters not present.
        4.3. Remove words with too many instacnes of a character.
        4.4. Filter for correct characters in correct positions.
        4.5. Filter for wrong characters in places other than as indicated in
             `wrong_place`.
        4.6. Score and sort words.
    5. Print and return tuple[(word, score)]
    """
    parser = ArgumentParser()
    parser.add_argument('-l', '--wlen', type=int, default=5)
    parser.add_argument('-c', '--correct', nargs='+')
    parser.add_argument('-p', '--wrong_place', nargs='+')
    parser.add_argument('-n', '--not_present')
    parser.add_argument('-g', '--n_guesses', type=int, default=5)
    parser.add_argument('-f', '--char_frequency', default='ENGLISH_DICT')
    parser.add_argument('-w', '--freq_weight', type=float, default=0.001)
    parser.add_argument('--vocab_file', default='corncob_caps.txt')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    with open(args.vocab_file) as f:
        all_words = tuple(f.read().upper().split('\n'))

    args = clean_input(args)

    filter_steps = (
        (filter_by_wlen, args.wlen),
        (remove_not_present, (args.correct, args.wrong_place, args.not_present)),
        (remove_maxed, (args.correct, args.wrong_place, args.not_present)),
        (filter_by_correct, args.correct),
        (filter_by_wrong_place, args.wrong_place),
        (sort_words, (args.char_frequency, args.freq_weight))
    )
    if args.verbose:
        print('Init:')
        print(all_words[:10])
    for step, arg in filter_steps:
        if args.verbose:
            print(f'{step.__name__}:')
        if arg:
            all_words = step(all_words, arg)
            if not all_words:
                print('Ran out of words')
                return tuple()
            if args.verbose:
                print(all_words[:10])

    guess_words = all_words[:args.n_guesses]

    print('Guess: Score')
    for guess, score in guess_words:
        print(f'{guess}: {score}')

    return guess_words


if __name__ == '__main__':
    main()
