import os
import pickle
import time
import argparse
import zipfile
import urllib.request

import numpy as np

from src.preprocessing import preprocess, skipgram_pairs
from src.model import Word2Vec

DATASET_URL = "http://mattmahoney.net/dc/text8.zip"
DATASET_ZIP = "text8.zip"
DATASET_FILE = "text8"

def download_text8(save_dir: str = "../data") -> str:
    """
    Download and extract the text8 dataset if it is not already present.

    :param save_dir: Directory where the files will be stored.
    :return: Path to the extracted text file.
    """
    os.makedirs(save_dir, exist_ok=True)

    zip_path = os.path.join(save_dir, DATASET_ZIP)
    txt_path = os.path.join(save_dir, DATASET_FILE)

    if os.path.exists(txt_path):
        print(f"Found existing dataset at {txt_path}")
        return txt_path

    if not os.path.exists(zip_path):
        print(f"Downloading text8 from {DATASET_URL}")
        urllib.request.urlretrieve(DATASET_URL, zip_path)
        print("Download complete")

    print("Extracting")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(save_dir)
    print("Extraction complete.")

    if os.path.exists(zip_path):
        os.remove(zip_path)
        print("ZIP file removed.")

    return txt_path

def iter_batches(center_ids: np.ndarray, context_ids: np.ndarray, batch_size: int):
    """
    Yield shuffled (center_batch, context_batch) pairs of size batch_size.

    The last batch may be smaller than batch_size if the total number of
    pairs is not divisible by batch_size.

    :param center_ids: Array of center word indices.
    :param context_ids: Array of context word indices (same length).
    :param batch_size: Number of pairs per batch.
    """
    n = len(center_ids)
    indices = np.random.permutation(n)
    for start in range(0, n, batch_size):
        idx = indices[start : start + batch_size]
        yield center_ids[idx], context_ids[idx]

def train(
    text: str,
    epochs: int = 5,
    embedding_dim: int = 100,
    window_size: int = 5,
    n_negatives: int = 5,
    batch_size: int = 512,
    learning_rate: float = 0.025,
    min_lr: float = 0.0001,
    min_count: int = 5,
    phrase_passes: int = 0,
    subsample_threshold: float = 1e-4,
    save_path: str = "./embeddings/model.npz",
    log_every: int = 100_000,
) -> "Word2Vec":
    """
    Run the full Word2Vec training pipeline.

    :param text: Raw corpus string.
    :param epochs: Number of full passes over the training pairs.
    :param embedding_dim: Dimensionality of word vectors.
    :param window_size: Maximum context-window radius.
    :param n_negatives: Number of negative samples per positive pair.
    :param batch_size: Mini-batch size (number of skip-gram pairs).
    :param learning_rate: Initial learning rate.
    :param min_lr: Minimum learning rate; decay stops here.
    :param min_count: Words appearing fewer times are discarded from vocab.
    :param phrase_passes: How many rounds of bigram phrase detection to run.
    :param subsample_threshold: Subsampling threshold.
    :param save_path: Save the model to this .npz path after training.
    :param log_every: Logging per x steps.
    :return: Trained ``Word2Vec`` instance.
    """
    print("Preprocessing")
    vocab, tokens = preprocess(
        text,
        min_count = min_count,
        phrase_passes = phrase_passes,
        subsample_threshold = subsample_threshold
    )
    print(
        f"Vocab size: {len(vocab):,}  |  "
        f"Tokens after subsampling: {len(tokens):,}"
    )

    print("Building skip-gram pairs")
    token_arr = np.array(tokens)
    pairs = list(skipgram_pairs(token_arr, vocab.word2id, window_size = window_size))

    center_ids = np.array([p[0] for p in pairs], dtype = np.int32)
    context_ids = np.array([p[1] for p in pairs], dtype = np.int32)
    n_pairs = len(pairs)
    print(f"Skip-gram pairs: {n_pairs:,}")

    model = Word2Vec(vocab, embedding_dim = embedding_dim, n_negatives = n_negatives)

    if save_path:
        save_dir = os.path.dirname(save_path) or "."
        os.makedirs(save_dir, exist_ok=True)
        vocab_path = os.path.join(save_dir, "vocab.pkl")
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {vocab_path}")

    steps_per_epoch = (n_pairs + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    step = 0
    global_step = 0
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_pairs = 0
        t0 = time.time()

        log_loss = 0.0
        log_pairs = 0

        for center_batch, context_batch in iter_batches(
            center_ids, context_ids, batch_size
        ):
            lr = max(min_lr, learning_rate * (1.0 - step / total_steps))

            loss = model.train_step(center_batch, context_batch, lr)
            n  = len(center_batch)
            epoch_loss  += loss * n
            epoch_pairs += n
            log_loss += loss * n
            log_pairs += n

            step += 1
            global_step += len(center_batch)
            if global_step % log_every < batch_size:
                avg_log_loss = log_loss / log_pairs
                progress = 100 * step / total_steps
                print(
                    f"  [{progress:5.1f}%] step={step * batch_size:_}  "
                    f"loss={avg_log_loss:.4f}  lr={lr:.6f}"
                )
                log_loss = 0.0
                log_pairs = 0

        elapsed = time.time() - t0
        avg_loss = epoch_loss / epoch_pairs
        print(
            f"Epoch {epoch}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"lr={lr:.6f}  "
            f"time={elapsed:.1f}s"
        )
        if save_path:
            base_name, ext = os.path.splitext(save_path)
            if not ext:
                ext = ".npz"

            epoch_save_path = f"{base_name}_epoch{epoch}{ext}"
            model.save(epoch_save_path)
            print(f"Saved checkpoint: {epoch_save_path}\n")

    return model

# def main() -> None:
#     parser = argparse.ArgumentParser(
#         description = "Train Word2Vec."
#     )
#     parser.add_argument(
#         "--data",
#         type = str,
#         default = None,
#         help = "Path to a plain-text corpus. If omitted, text8 is downloaded automatically.",
#     )
#     parser.add_argument(
#         "--max_chars",
#         type = int,
#         default = None,
#         help = "Read at most this many characters from the corpus.",
#     )
#     parser.add_argument("--epochs", type = int, default = 5)
#     parser.add_argument("--embedding_dim", type = int, default = 100)
#     parser.add_argument("--window_size", type = int, default = 5)
#     parser.add_argument("--n_negatives", type = int, default = 5)
#     parser.add_argument("--batch_size", type = int, default = 512)
#     parser.add_argument("--lr", type = float, default = 0.025, dest = "learning_rate")
#     parser.add_argument("--min_lr", type = float, default = 0.0001)
#     parser.add_argument("--min_count", type = int, default = 5)
#     parser.add_argument("--subsample_threshold", type = float, default = 1e-4)
#     parser.add_argument(
#         "--save",
#         type = str,
#         default = "./embeddings/model.npz",
#         dest = "save_path",
#         help = "Path to save the trained embeddings.",
#     )
#     args = parser.parse_args()
#
#     data_path = args.data if args.data else download_text8()
#
#     with open(data_path, "r", encoding="utf-8") as fh:
#         text = fh.read(args.max_chars) if args.max_chars else fh.read()
#
#     train(
#         text = text,
#         epochs = args.epochs,
#         embedding_dim=  args.embedding_dim,
#         window_size = args.window_size,
#         n_negatives = args.n_negatives,
#         batch_size = args.batch_size,
#         learning_rate = args.learning_rate,
#         min_lr = args.min_lr,
#         min_count = args.min_count,
#         subsample_threshold = args.subsample_threshold,
#         save_path = args.save_path,
#     )
#
# if __name__ == "__main__":
#     main()