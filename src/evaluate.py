import numpy as np
from src.model import Word2Vec
import os
import urllib.request
import zipfile
import pickle

GOOGLE_ANALOGY_URL = "https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip"
ANALOGY_FILE = "questions-words.txt"


def download_google_analogy(save_dir: str = "./data") -> str:
    """
    Download the Google analogy benchmark (questions-words.txt) if not present.
    Returns path to the file.
    """
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, ANALOGY_FILE)

    if os.path.exists(file_path):
        print(f"Found existing file at {file_path}")
        return file_path

    print(f"Downloading Google analogy dataset from {GOOGLE_ANALOGY_URL}")
    zip_path = os.path.join(save_dir, "word2vec_source.zip")
    urllib.request.urlretrieve(GOOGLE_ANALOGY_URL, zip_path)
    print("Download complete. Extracting.")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("questions-words.txt"):
                with zf.open(member) as source, open(file_path, "wb") as target:
                    target.write(source.read())
                break

    if os.path.exists(zip_path):
        os.remove(zip_path)
    print(f"Dataset ready at {file_path}")
    return file_path


def get_embedding(model: Word2Vec, word: str) -> np.ndarray:
    if word not in model.vocab:
        raise KeyError(f"'{word}' not in vocabulary.")
    return model.W[model.vocab.word2id[word]]

def cosine_similarity(model: Word2Vec, word_a: str, word_b: str) -> float:
    va = get_embedding(model, word_a)
    vb = get_embedding(model, word_b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-10))

def most_similar(model: Word2Vec, word: str, topk: int = 10) -> list[tuple[str, float]]:
    vec = get_embedding(model, word)
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    norms = np.linalg.norm(model.W, axis=1, keepdims=True) + 1e-10
    sims = (model.W / norms) @ vec

    sims[model.vocab.word2id[word]] = -np.inf

    top_ids = np.argpartition(sims, -topk)[-topk:]
    top_ids = top_ids[np.argsort(sims[top_ids])[::-1]]

    return [(model.vocab.id2word[i], float(sims[i])) for i in top_ids]


def analogy(model: Word2Vec, a: str, b: str, c: str,
            topk: int = 5, W_norm: np.ndarray = None) -> list[tuple[str, float]]:
    for word in (a, b, c):
        if word not in model.vocab:
            raise KeyError(f"'{word}' not in vocabulary.")

    if W_norm is None:
        norms  = np.linalg.norm(model.W, axis=1, keepdims=True) + 1e-10
        W_norm = model.W / norms

    va = W_norm[model.vocab.word2id[a]]
    vb = W_norm[model.vocab.word2id[b]]
    vc = W_norm[model.vocab.word2id[c]]

    cos_a = (W_norm @ va + 1) / 2
    cos_b = (W_norm @ vb + 1) / 2
    cos_c = (W_norm @ vc + 1) / 2
    scores = (cos_b * cos_c) / (cos_a + 1e-10)

    for word in (a, b, c):
        scores[model.vocab.word2id[word]] = -np.inf

    top_ids = np.argpartition(scores, -topk)[-topk:]
    top_ids = top_ids[np.argsort(scores[top_ids])[::-1]]

    return [(model.vocab.id2word[i], float(scores[i])) for i in top_ids]


def evaluate_analogy_file(model: Word2Vec, path: str) -> dict:
    results: dict[str, dict] = {}
    current_category = "uncategorized"
    total = correct = skipped = 0

    print("Precomputing normalized embedding matrix")
    norms = np.linalg.norm(model.W, axis=1, keepdims=True) + 1e-10
    W_norm = model.W / norms

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"Found {total_lines} total lines.")

    for i, line in enumerate(lines):
        if i > 0 and i % 1000 == 0:
            print(f" Progress: {i}/{total_lines} ({(i/total_lines)*100:.1f}%)")

        line = line.strip().lower()
        if not line:
            continue

        if line.startswith(":"):
            current_category = line[2:].strip()
            results[current_category] = {"correct": 0, "total": 0, "skipped": 0}
            continue

        parts = line.split()
        if len(parts) != 4:
            continue

        a, b, c, expected = parts

        if any(w not in model.vocab for w in (a, b, c, expected)):
            results[current_category]["skipped"] += 1
            skipped += 1
            continue

        total += 1
        results[current_category]["total"] += 1

        try:
            predicted = analogy(model, a, b, c, topk=1, W_norm=W_norm)[0][0]
        except Exception:
            continue

        if predicted == expected:
            correct += 1
            results[current_category]["correct"] += 1

    print(f"\n{'Category':<35} {'Acc':>6}  {'Correct':>7}  {'Total':>7}  {'Skipped':>7}")
    print("─" * 70)
    for cat, s in results.items():
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {cat:<33} {acc:>5.1%}  {s['correct']:>7}  {s['total']:>7}  {s['skipped']:>7}")

    overall = correct / total if total > 0 else 0.0
    print("─" * 70)
    print(f"  {'TOTAL':<33} {overall:>5.1%}  {correct:>7}  {total:>7}  {skipped:>7}\n")

    return {
        "overall_accuracy": overall,
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "categories": results,
    }


def run_evaluation(model_path: str):
    """
    Loads vocabulary and model, download test dataset and runs evaluation.
    """
    # import sys
    # import src.preprocessing
    # sys.modules['preprocessing'] = src.preprocessing
    model_dir = os.path.dirname(model_path)
    vocab_path = os.path.join(model_dir, "vocab.pkl")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"No vocab found here: {vocab_path}")

    print(f"Loading vocab {vocab_path}")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    print(f"Loading model from {model_path}")
    model = Word2Vec(vocab)

    model.load(model_path)

    print("Downloading evaluation data")
    dataset_path = download_google_analogy(save_dir="./data")

    print("Starting evaluation")
    evaluate_analogy_file(model, dataset_path)