# word2vec

Pure-NumPy implementation of word2vec using the skip-gram with negative sampling variant.

---

## Results

Evaluated on the [Google word analogy benchmark](https://code.google.com/archive/p/word2vec/). The model was trained on the [text8](http://mattmahoney.net/dc/text8.zip) corpus with default hyperparameters but 15 epochs (see table below).

| Category                    | Acc       | Correct  | Total     | Skipped  |
|-----------------------------|-----------|----------|-----------|----------|
| capital-common-countries    | 72.9%     | 369      | 506       | 0        |
| capital-world               | 37.5%     | 1337     | 3564      | 960      |
| currency                    | 14.4%     | 86       | 596       | 270      |
| city-in-state               | 30.0%     | 700      | 2330      | 137      |
| family                      | 36.0%     | 151      | 420       | 86       |
| gram1-adjective-to-adverb   | 7.4%      | 73       | 992       | 0        |
| gram2-opposite              | 5.4%      | 41       | 756       | 56       |
| gram3-comparative           | 44.1%     | 587      | 1332      | 0        |
| gram4-superlative           | 15.6%     | 155      | 992       | 130      |
| gram5-present-participle    | 16.7%     | 176      | 1056      | 0        |
| gram6-nationality-adjective | 76.4%     | 1162     | 1521      | 78       |
| gram7-past-tense            | 22.3%     | 348      | 1560      | 0        |
| gram8-plural                | 35.5%     | 473      | 1332      | 0        |
| gram9-plural-verbs          | 19.5%     | 170      | 870       | 0        |
| **TOTAL**                   | **32.7%** | **5828** | **17827** | **1717** |

---

## Architecture

### Skip-gram with negative sampling

For each center word $w$ and context word $c$ drawn from a window of radius $r$, the model maximises:

$$J = \log \sigma(v_w \cdot u_c) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n}\bigl[\log \sigma(-v_w \cdot u_{w_k})\bigr]$$

where $v_w \in \mathbb{R}^d$ is the center embedding, $u_c \in \mathbb{R}^d$ is the context embedding, $K$ is the number of negative samples, and $P_n(w) \propto \text{count}(w)^{3/4}$ is the noise distribution.

### Gradients

Let $\sigma_+ = \sigma(v_w \cdot u_c)$ and $\sigma_k = \sigma(v_w \cdot u_{w_k})$:

$$\frac{\partial \mathcal{L}}{\partial u_c} = (\sigma_+ - 1)\, v_w \qquad \frac{\partial \mathcal{L}}{\partial u_{w_k}} = \sigma_k\, v_w \qquad \frac{\partial \mathcal{L}}{\partial v_w} = (\sigma_+ - 1)\, u_c + \sum_{k=1}^{K} \sigma_k\, u_{w_k}$$

Updates follow SGD with linear learning rate decay over training.

---

## Preprocessing pipeline

1. **Clean** - lowercase, strip non-alphabetic characters.
2. **Phrase detection** - merge frequent bigrams into single tokens (e.g. `new york` → `new_york`) using the scoring formula from Mikolov et al. (2013); repeated for `phrase_passes` iterations.
3. **Vocabulary** - retain only words with frequency ≥ `min_count`; compute $P_n$ for negative sampling.
4. **Subsampling** - discard high-frequency tokens with probability $1 - \sqrt{t / f(w)}$.
5. **Skip-gram pairs** - slide a dynamic window over the subsampled sequence.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # Unix
pip install -r requirements.txt
```

---

## Usage

### Train on text8 (downloaded automatically)

```bash
python main.py train
```

### Train on a custom corpus

```bash
python main.py train --data /path/to/corpus.txt \
    --epochs 5 \
    --embedding_dim 100 \
    --window_size 5 \
    --n_negatives 5 \
    --batch_size 512 \
    --lr 0.025 \
    --min_lr 0.0001 \
    --min_count 5 \
    --subsample_threshold 1e-4 \
    --phrase_passes 0 \
    --save ./embeddings/model.npz
```

### Evaluate on the Google analogy benchmark

```bash
python main.py evaluate --model_path ./embeddings/model_epoch5.npz
```

The benchmark file is downloaded automatically on first run.

---

## Hyperparameters

| Parameter             | Default | Notes                                                                     |
|-----------------------|---------|---------------------------------------------------------------------------|
| `embedding_dim`       | 100     | Vector dimensionality                                                     |
| `window_size`         | 5       | Maximum context-window radius (actual radius sampled in [1, window_size]) |
| `n_negatives`         | 5       | Negative samples per positive pair; 5–20 recommended for large corpora    |
| `batch_size`          | 512     | Skip-gram pairs per gradient step                                         |
| `learning_rate`       | 0.025   | Initial LR; linearly decayed to `min_lr` over training                    |
| `min_lr`              | `learning_rate × 1e-4` | Minimum learning rate (Mikolov's `starting_alpha × 0.0001`)     |
| `min_count`           | 5       | Words below this frequency are excluded from the vocabulary               |
| `subsample_threshold` | 1e-4    | Controls how aggressively frequent words are discarded                    |

Based on:

- [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546)
- [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781)
- [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722)