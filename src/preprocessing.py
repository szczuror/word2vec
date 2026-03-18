import re
import numpy as np
from collections import Counter

# tokenization, text cleaning, vocab building
# dataset I chose already has all small letters and no interpunction signs, however, I will provide a fucntion to do so
# in case i want to use another dataset

regex_pattern = re.compile(r"[^a-z'\s]")
def clean(text: str) -> list[str]:
    """
    Cleans the input text by lowering case, removing punctuation, and splitting into tokens.
    :param text: Raw input string.
    :return: List of cleaned word tokens.
    """
    return regex_pattern.sub(' ', text.lower()).split()

def build_vocab(tokens: list[str]) -> tuple[dict[str, int], dict[int, str], Counter]:
    """
    Builds vocabulary mappings and counts word frequencies.
    :param tokens: List of all tokens.
    :return: A tuple of (word2id, id2word, word_counts).
    """
    word_counts = Counter(tokens)
    words = sorted(list(word_counts.keys()))

    word2id = {word: i for i, word in enumerate(words)}
    id2word = {i: word for i, word in enumerate(words)}

    return word2id, id2word, word_counts

def skipgram_pairs(tokens: list[str], word2id: dict[str, int], window_size: int = 2):
    """
    Generator for (center_word, context_id) pairs using a dynamic window size.
    :param tokens: List of tokens.
    :param word2id: Dictionary mapping words to their IDs.
    :param window_size: Maximum radius of the context window.
    """
    n = len(tokens)

    dynamic_windows = np.random.randint(1, window_size + 1, size = n)

    for i, center in enumerate(tokens):
        if center not in word2id:
            continue
        window = dynamic_windows[i]
        start = max(0, i - window)
        end = min(n, i + window + 1)

        for j in range(start, end):
            if i != j:
                yield center, word2id[tokens[j]]


def get_negative_sampling_distribution(word_counts: Counter, word2id: dict[str, int]) -> np.ndarray:
    """
    Calculates the probability distribution for Negative Sampling.
    Uses the formula P(w) = count(w)^0.75 / sum(count^0.75) which helps boost rare words.
    :param word_counts: Counter object with word frequencies.
    :param word2id: Dictionary mapping words to IDs.
    :return: Numpy array of probabilities for each word ID.
    """
    vocab_size = len(word2id)

    counts_array = np.zeros(vocab_size)

    for word, count in word_counts.items():
        if word in word2id:
            word_id = word2id[word]
            counts_array[word_id] = count

    p_n = np.power(counts_array, 0.75)

    p_n = p_n / np.sum(p_n)

    return p_n

def detect_phrases(tokens: list[str], word_counts: Counter, delta: float = 5.0, threshold: float = 1e-4) -> list[str]:
    """
    Identifies frequent bigrams and merges them into single tokens.
    :param tokens: List of tokens.
    :param word_counts: Counter object with word frequencies.
    :param delta: Discounting factor to prevent forming phrases from very rare words.
    :param threshold: Significance threshold; higher values result in fewer, more certain phrases.
    :return: New list of tokens with merged phrases.
    """
    bigram_counts = Counter(zip(tokens[:-1], tokens[1:]))
    
    new_tokens = []
    i = 0
    n = len(tokens)
    
    while i < n:
        if i < n - 1:
            w1 = tokens[i]
            w2 = tokens[i+1]
            bigram = (w1, w2)
            
            c_w1_w2 = bigram_counts[bigram]
            
            if c_w1_w2 > delta:
                c_w1 = word_counts[w1]
                c_w2 = word_counts[w2]
                
                score = ((c_w1_w2 - delta) / (c_w1 * c_w2)) * n
                
                if score > threshold:
                    new_tokens.append(f"{w1}_{w2}")
                    i += 2
                    continue
        
        new_tokens.append(tokens[i])
        i += 1
        
    return new_tokens

def subsampling(tokens: list[str], word_counts: Counter, min_count: int = 2, threshold: float = 1e-4) -> list[str]:
    """
    Randomly discards frequent words to speed up training and improve the quality of rare word vectors.
    :param tokens: List of tokens.
    :param word_counts: Counter object with word frequencies.
    :param min_count: Words appearing fewer times than this are discarded.
    :param threshold: Subsampling threshold (usually between 1e-3 and 1e-5).
    :return: Filtered list of tokens.
    """
    words_len = len(tokens) # ;)

    keep_probs = {}
    for word, count in word_counts.items():
        if count < min_count:
            keep_probs[word] = 0.0
        else:
            freq = count / words_len
            # p_drop = 1 - sqrt(threshold / freq), 
            # p_keep = 1 - p_drop = sqrt(threshold / freq)
            keep_probs[word] = min(1.0, np.sqrt(threshold / freq))
    
    random_vec = np.random.rand(words_len)

    filtered = [word for i, word in enumerate(tokens)
        if keep_probs.get(word, 0.0) > random_vec[i]
    ]

    return filtered

def build_phrases_multi_pass(tokens: list[str], passes: int = 3, start_threshold: float = 1e-4, delta: float = 5.0) -> list[str]:
    """
    Runs phrase detection multiple times to capture longer n-grams.
    The threshold is lowered each pass to capture progressively more complex phrases.
    :param tokens: List of input tokens.
    :param passes: Number of iterations.
    :param start_threshold: Initial threshold for detect_phrases.
    :param delta: Delta parameter for detect_phrases.
    :return: List of tokens after phrase merging.
    """
    current_tokens = tokens
    current_threshold = start_threshold

    for p in range(passes):
        # print(f"Pass {p+1}/{passes} (threshold: {current_threshold:.6f})")
        
        current_word_counts = Counter(current_tokens)
        
        current_tokens = detect_phrases(
            tokens=current_tokens, 
            word_counts=current_word_counts, 
            delta=delta, 
            threshold=current_threshold
        )
        
        current_threshold /= 2.0

    return current_tokens