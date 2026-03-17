# tokenization, text cleaning, vocab building
# dataset I chose already has all small letters and no interpunction signs, however, I will provide a fucntion to do so
# in case i want to use another dataset
import re


def clean(text: str) -> list[str]:
    """
    Function to make all words lowercase and remove punctuation
    :param text:
    :return:
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()

    return tokens

def build_vocabulary(tokens: list[str]) -> tuple[dict[str, int], dict[int, str], int]:
    """
    Function to build vocabulary from tokens
    :param tokens:
    :return:
    """
    words = sorted(list(set(tokens)))

    word2id = {word: i for i, word in enumerate(words)}
    id2word = {i: word for i, word in enumerate(words)}

    vocab_size = len(words)

    return word2id, id2word, vocab_size

def skipgram_pairs(tokens: list[str], word2id: dict[str, int], window_size: int = 2) -> list[tuple[int, int]]:
    """
    Function to build pairs of skip-gram tokens
    :param tokens:
    :param word2id:
    :param window_size:
    :return:
    """
    pairs = list[tuple[int, int]] = []
    n: int = len(tokens)

    for i, word in enumerate(tokens):
        center = word2id[word]

        start = max(0, i - window_size)
        end = min(n, i + window_size + 1)

        for j in range(start, end):
            if i != j:
                pairs.append((center, word2id[tokens[j]]))

    return pairs