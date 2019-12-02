import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, num_samples: int = 0, num_features: int = 0):
        self.best_image2text_recall_at_k = (-1.0, -1.0, -1.0)
        self.cur_image2text_recall_at_k = (-1.0, -1.0, -1.0)
        self.best_text2image_recall_at_k = (-1.0, -1.0, -1.0)
        self.cur_text2image_recall_at_k = (-1.0, -1.0, -1.0)
        self.index_update = 0
        self.num_samples = num_samples
        self.num_features = num_features
        self.embedded_images = np.zeros((self.num_samples, self.num_features))
        self.embedded_sentences = np.zeros((self.num_samples, self.num_features))

    def reset_all_vars(self) -> None:
        self.index_update = 0
        self.embedded_images = np.zeros((self.num_samples, self.num_features))
        self.embedded_sentences = np.zeros((self.num_samples, self.num_features))
        self.cur_text2image_recall_at_k = (-1.0, -1.0, -1.0)
        self.cur_image2text_recall_at_k = (-1.0, -1.0, -1.0)

    def update_embeddings(
        self, embedded_images: np.ndarray, embedded_sentences: np.ndarray
    ) -> None:
        num_samples = embedded_images.shape[0]
        self.embedded_images[
            self.index_update : self.index_update + num_samples, :
        ] = embedded_images
        self.embedded_sentences[
            self.index_update : self.index_update + num_samples, :
        ] = embedded_sentences
        self.index_update += num_samples

    def is_best_recall_at_k(self) -> bool:
        # Update current
        self.cur_image2text_recall_at_k = self.image2text_recall_at_k()
        self.cur_text2image_recall_at_k = self.text2image_recall_at_k()
        # Sum recalls
        image2text_recall_at_ks = sum(self.cur_image2text_recall_at_k)
        text2image_recall_at_ks = sum(self.cur_text2image_recall_at_k)
        # Sum best recalls
        best_image2text_recall_at_ks = sum(self.best_image2text_recall_at_k)
        best_text2image_recall_at_ks = sum(self.best_text2image_recall_at_k)
        # Check if the current are the better
        if (image2text_recall_at_ks + text2image_recall_at_ks) > (
            best_image2text_recall_at_ks + best_text2image_recall_at_ks
        ):
            return True
        return False

    def update_best_recall_at_k(self):
        self.best_image2text_recall_at_k = self.cur_image2text_recall_at_k
        self.best_text2image_recall_at_k = self.cur_text2image_recall_at_k

    def image2text_recall_at_k(self):
        """Computes the recall at K when doing image to text retrieval and updates the
        object variable.

        Returns:
            The recall at 1, 5, 10.

        """
        num_images = self.embedded_images.shape[0]
        ranks = np.zeros(num_images)
        for index in range(num_images):
            # Get query image
            query_image = self.embedded_images[index]
            # Similarities
            similarities = np.dot(query_image, self.embedded_sentences.T).flatten()
            indices = np.argsort(similarities)[::-1]
            # Score
            tmp = np.where(indices == index)[0][0]
            ranks[index] = tmp

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        return r1, r5, r10

    def text2image_recall_at_k(self):
        """Computes the recall at K when doing image to text retrieval and updates the
        object variable.
        Returns:
            The recall at 1, 5, 10.

        """
        num_captions = self.embedded_images.shape[0]
        ranks = np.zeros(num_captions)
        for index in range(num_captions):
            # Get query image
            query_sentence = self.embedded_sentences[index]
            # Similarities
            similarities = np.dot(query_sentence, self.embedded_images.T).flatten()
            indices = np.argsort(similarities)[::-1]
            # Score
            tmp = np.where(indices == index)[0][0]
            ranks[index] = tmp

        r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        return r1, r5, r10
