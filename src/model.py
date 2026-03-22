import numpy as np
from src.preprocessing import Vocabulary

class Word2Vec:
    """
    Word2Vec model
    """
    def __init__(self, vocab: Vocabulary, embedding_dim: int = 100, n_negatives: int = 5, max_neg_resample_iters: int = 10):
        """
        :param vocab: Vocabulary object (provides size and neg sampling probs).
        :param embedding_dim: Dimensionality of word vectors.
        :param n_negatives: Number of negative samples per positive pair (k).
        """
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.n_negatives = n_negatives
        self.max_neg_resample_iters = max_neg_resample_iters

        vocab_size = len(vocab)

        self.W = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))
        self.W_ = np.zeros((vocab_size, self.embedding_dim))

    @staticmethod
    def _sigmoid(x : np.ndarray) -> np.ndarray:
        """
        Sigmoid function with clipping to avoid overflow in exp(x).
        :param x: Input array.
        :return: Array of sigmoid values.
        """
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _sample_negatives(self, context_ids: np.ndarray, center_ids: np.ndarray) -> np.ndarray:
        """
        Draw k negative samples from the noise distribution P^(3/4).
        Resamples until we have exactly k samples that are neither the
        true context word nor the center word.
        :param context_ids: IDs of true context words (excluded from negatives).
        :param center_ids: IDs of center words (excluded from negatives).
        :return: Array of negative sample IDs
        """
        samples = np.random.choice(
            len(self.vocab),
            size = (len(center_ids), self.n_negatives),
            p = self.vocab.neg_sampling_probs
        )

        for _ in range(self.max_neg_resample_iters):
            collisions = (
            (samples == center_ids[:, None]) | (samples == context_ids[:, None])
            )

            if not collisions.any():
                break

            samples[collisions] = np.random.choice(
                len(self.vocab),
                size = collisions.sum(),
                p = self.vocab.neg_sampling_probs
            )

        return samples

    def forward(self, v_w: np.ndarray, u_c: np.ndarray, u_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass using pre-fetched embeddings.
        :param v_w: Center word embeddings
        :param u_c: Context word embeddings
        :param u_neg: Negative sample embeddings
        :return: sig_pos, sig_neg
        """
        sig_pos = self._sigmoid(np.einsum('bd,bd->b', v_w, u_c))
        sig_neg = self._sigmoid(np.einsum('bd,bkd->bk', v_w, u_neg))

        return sig_pos, sig_neg

    @staticmethod
    def loss(sig_pos: np.ndarray, sig_neg: np.ndarray) -> float:
        """
        Compute the negative sampling loss for a batch of center-context pairs.

        :param sig_pos: Sigmoid values for positive pairs
        :param sig_neg: Sigmoid values for negative samples
        :return: Mean loss as float
        """
        eps = 1e-15
        pos_loss = -np.log(sig_pos + eps)
        neg_loss = -np.sum(np.log(1 - sig_neg + eps), axis=1)

        return float(np.mean(pos_loss + neg_loss))

    @staticmethod
    def backward(v_w: np.ndarray, u_c: np.ndarray, u_neg: np.ndarray,
             sig_pos: np.ndarray, sig_neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gradients for skip-gram with negative sampling.

        :param v_w: Center word embeddings
        :param u_c: True context word embeddings
        :param u_neg: Negative word embeddings
        :param sig_pos: Sigmoid of dot product for positive pairs
        :param sig_neg: Sigmoid of dot product for negative pairs
        :return: Tuple of gradients (grad_v_w, grad_u_c, grad_u_neg)
        """
        e_pos = sig_pos - 1
        e_neg = sig_neg
        batch_size = v_w.shape[0]

        grad_u_c = np.einsum('b,bd->bd', e_pos, v_w)
        grad_u_neg = np.einsum('bk,bd->bkd', e_neg, v_w)
        grad_v_w = (np.einsum('b,bd->bd', e_pos, u_c)
                    + np.einsum('bk,bkd->bd', e_neg, u_neg))

        return grad_v_w / batch_size, grad_u_c / batch_size, grad_u_neg / batch_size

    def update(self, center_ids: np.ndarray, context_ids: np.ndarray, neg_ids: np.ndarray,
               grad_v_w: np.ndarray, grad_u_c: np.ndarray, grad_u_neg: np.ndarray,
               lr: float) -> None:
        """
        Update parameters for all three embedding matrices.

        :param center_ids: Center word indices
        :param context_ids: True context word indices
        :param neg_ids: Negative sample indices
        :param grad_v_w: Gradients for center embeddings
        :param grad_u_c: Gradients for context embeddings
        :param grad_u_neg: Gradients for negative embeddings
        :param lr: Learning rate
        """
        np.add.at(self.W, center_ids, -lr * grad_v_w)
        np.add.at(self.W_, context_ids, -lr * grad_u_c)

        flat_neg_ids = neg_ids.reshape(-1)
        flat_neg_grads = grad_u_neg.reshape(-1, self.embedding_dim)
        np.add.at(self.W_, flat_neg_ids, -lr * flat_neg_grads)

    def train_step(self, center_ids: np.ndarray, context_ids: np.ndarray, lr: float) -> float:
        """
        Train the model for each three embedding matrices.

        :param center_ids: Center word indices
        :param context_ids: True context word indices
        :param lr: Learning rate
        :return: Mean loss as float over the batch
        """

        neg_ids = self._sample_negatives(context_ids, center_ids)

        v_w = self.W[center_ids]
        u_c = self.W_[context_ids]
        u_neg = self.W_[neg_ids]

        sig_pos, sig_neg = self.forward(v_w, u_c, u_neg)

        curr_loss = self.loss(sig_pos, sig_neg)

        grad_v_w, grad_u_c, grad_u_neg = self.backward(v_w, u_c, u_neg,sig_pos, sig_neg)

        self.update(center_ids, context_ids, neg_ids, grad_v_w, grad_u_c, grad_u_neg, lr)

        return curr_loss

    def save(self, path: str) -> None:
        """
        Save both embedding matrices to a .npz file
        :param path: Destination path
        """
        np.savez(path, W = self.W, W_ = self.W_)

    def load(self, path: str) -> "Word2Vec":
        """
        Load embedding matrices from a .npz file
        :param path: Path to the .npz file
        :return: self
        """
        data = np.load(path)
        self.W = data["W"]
        self.W_ = data["W_"]
        self.embedding_dim = self.W.shape[1]
        return self