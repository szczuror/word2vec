import re
import numpy as np
from collections import Counter

# tokenization, text cleaning, vocab building
# dataset I chose already has all small letters and no interpunction signs, however, I will provide a fucntion to do so
# in case i want to use another dataset

regex_pattern = re.compile(r"[^a-z'\s]")
def clean(text: str) -> list[str]:
    """
    Function to make all words lowercase and remove punctuation
    :param text:
    :return:
    """
    return regex_pattern.sub(' ', text.lower()).split()

def build_vocab(tokens: list[str]) -> tuple[dict[str, int], dict[int, str], Counter]:
    """
    Function to build vocabulary from tokens
    :param tokens:
    :return:
    """
    word_counts = Counter(tokens)
    words = sorted(list(word_counts.keys()))

    word2id = {word: i for i, word in enumerate(words)}
    id2word = {i: word for i, word in enumerate(words)}

    return word2id, id2word, word_counts

def skipgram_pairs(tokens: list[str], word2id: dict[str, int], window_size: int = 2):
    """
    Function to build pairs of skip-gram tokens dynamically
    :param tokens:
    :param word2id:
    :param window_size:
    :return:
    """
    token_ids = [word2id[w] for w in tokens if w in word2id]
    n = len(token_ids)

    dynamic_windows = np.random.randint(1, window_size + 1, size = n)

    pairs = []
    for i, center in enumerate(token_ids):
        window = dynamic_windows[i]
        start = max(0, i - window)
        end = min(n, i + window + 1)

        for j in range(start, end):
            if i != j:
                yield (center, token_ids[j])


def get_negative_sampling_distribution(word_counts: Counter, word2id: dict[str, int]) -> np.ndarray:
    """
    Function to get negative sampling distribution
    """
    vocab_size = len(word2id)

    counts_array = np.zeros(vocab_size)

    for word, count in word_counts.items():
        if word in word2id:
            word_id = word2id[word]
            counts_array[word_id] = count

    p_n = np.power(counts_array, 0.75) # word2vec publication research

    p_n = p_n / np.sum(p_n)

    return p_n

def detect_phrases(tokens: list[str], word_counts: Counter, delta: float = 5.0, threshold: float = 1e-4) -> list[str]:
    """
    Detecting frequently appearing phrases and merging them
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
    current_tokens = tokens
    current_threshold = start_threshold

    for p in range(passes):
        print(f"Tworzenie fraz - Przebieg {p+1}/{passes} (próg: {current_threshold:.6f})...")
        
        current_word_counts = Counter(current_tokens)
        
        current_tokens = detect_phrases(
            tokens=current_tokens, 
            word_counts=current_word_counts, 
            delta=delta, 
            threshold=current_threshold
        )
        
        current_threshold /= 2.0

    return current_tokens

# TODO normalization? for example: cat, cats, and so on as one word. Doesnt seem like a very good idea tho