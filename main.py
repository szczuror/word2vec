from collections import Counter
from src.preprocessing import *

if __name__ == "__main__":
    with open("data/text8", 'r') as file:
        sample_text = file.read()
        # sample_text = sample_text[:10000000]

    print("Cleaning")
    tokens = clean(sample_text)

    print("Building phrases")
    tokens_with_phrases = build_phrases_multi_pass(tokens, passes=3, start_threshold=1e-4)

    print("\Searching merged for debug...")
    merged_phrases = [token for token in tokens_with_phrases if '_' in token]
    
    phrase_counts = Counter(merged_phrases)
    
    print(f"Znaleziono {len(phrase_counts)} unikalnych fraz.")
    print("Top 30 najpopularniejszych fraz:")
    
    for phrase, count in phrase_counts.most_common(30):
        print(f"{phrase}: {count}")
    
    print("Building vocab")
    word2id, id2word, word_counts = build_vocab(tokens_with_phrases)
    print(f"Vocab size: {len(word2id)}")

    print("Subsampling")
    tokens_sub = subsampling(tokens_with_phrases, word_counts, min_count=1)

    print("Pairing")
    pairs_gen = skipgram_pairs(tokens_sub, word2id, window_size=2)

    # 6. Negative sampling distribution
    p_n = get_negative_sampling_distribution(word_counts, word2id)
    
    print("\nDone.")