from sentence_transformers import SentenceTransformer
from typing import List, Union


def vectors_get_embedding_minilm(text: Union[List[str], str]) -> List[float]:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(text).tolist()
