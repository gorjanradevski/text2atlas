import numpy as np


class EmbeddedDoc:
    def __init__(
        self, doc_id: int, organ_indices: np.ndarray, doc_embedding: np.ndarray
    ):
        self.doc_id = doc_id
        self.organ_indices = organ_indices
        self.doc_embedding = doc_embedding

    def docs_distance(self, other):
        return np.linalg.norm(self.doc_embedding - other.doc_embedding, axis=-1)
