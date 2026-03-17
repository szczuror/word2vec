import re
import numpy as np
from collections import Counter

# tokenization, text cleaning, vocab building
# dataset I chose already has all small letters and no interpunction signs, however, I will provide a fucntion to do so
# in case i want to use another dataset


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

def subsampling(tokens: list[str], min_count: int = 2, threshold: float = 1e-4) -> list[str]:
    word_counts = Counter(tokens)
    words_len = len(tokens)

    filtered = []
    for word in tokens:
        count = word_counts[word]

        if count < min_count:
            continue

        freq = count / words_len
        
        p_drop = 1 - np.sqrt(threshold / freq)

        if p_drop > np.random.rand():
            continue

        filtered.append(word)
    
    return filtered


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
    Function to build pairs of skip-gram tokens dynamically
    :param tokens:
    :param word2id:
    :param window_size:
    :return:
    """
    pairs: list[tuple[int, int]] = []
    n: int = len(tokens)

    for i, word in enumerate(tokens):
        center = word2id[word]

        dynamic_window = np.random.randint(1, window_size + 1)
        start = max(0, i - dynamic_window)
        end = min(n, i + dynamic_window + 1)

        for j in range(start, end):
            if i != j:
                pairs.append((center, word2id[tokens[j]]))

    return pairs


def get_negative_sampling_distribution(tokens : list[str], word2id: dict[str, int]) -> np.ndarray:
    """
    Function to get negative sampling distribution
    """
    word_counts = Counter(tokens)
    vocab_size = len(word2id)

    counts_array = np.zeros(vocab_size)

    for word, count in word_counts.items():
        if word in word2id:
            word_id = word2id[word]
            counts_array[word_id] = count

    p_n = np.power(counts_array, 0.75) # word2vec publication research

    p_n = p_n / np.sum(p_n)

    return p_n

# TODO normalization? for example: cat, cats, and so on as one word. Doesnt seem like a very good idea tho