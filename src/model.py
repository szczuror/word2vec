import numpy as np
from preprocessing import Vocabulary

class Word2Vec:
    """
    Word2Vec model
    """
    def __init__(self, vocab: Vocabulary, embedding_dim: int = 100, n_negatives: int = 5):
        """
        :param vocab: Vocabulary object (provides size and neg sampling probs).
        :param embedding_dim: Dimensionality of word vectors.
        :param n_negatives: Number of negative samples per positive pair (k).
        """
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.n_negatives = n_negatives

        vocab_size = len(vocab)

        self.W = np.random.uniform(-0.5/embedding_dim, 0.5/embedding_dim, (vocab_size, embedding_dim))
        self.W_ = np.zeros((vocab_size, self.embedding_dim))

    @staticmethod
    def _sigmoid(x : np.ndarray) -> np.ndarray:
        """
        Sigmoid function with clipping to avoid overflow in exp(x).
        """
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _sample_negatives(self, context_ids: np.ndarray, center_ids: np.ndarray) -> np.ndarray:
        """
        Draw k negative samples from the noise distribution P^(3/4).
        Resamples until we have exactly k samples that are neither the
        true context word nor the center word.
        :param context_id: ID of the true context word (excluded from negatives).
        :param center_id: ID of the center word (excluded from negatives).
        :return: Array of k negative sample IDs

        """
        samples = np.random.choice(
            len(self.vocab),
            size = (len(center_ids), self.n_negatives),
            p = self.vocab.neg_sampling_probs
        )

        while True:
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

    def forward(self, center_ids: np.ndarray, context_ids: np.ndarray, neg_ids: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        v_w = self.W[center_ids]
        u_c = self.W_[context_ids]
        u_neg = self.W_[neg_ids]

        sig_pos = self._sigmoid(np.einsum('bd,bd->b', v_w, u_c))
        sig_neg = self._sigmoid(np.einsum('bd,bkd->bk', v_w, u_neg))

        return v_w, sig_pos, sig_neg