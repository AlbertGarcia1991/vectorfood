"""
This module constains all functions and objects to generate the embeddings
of different elements.
"""

from typing import List, Union

from sentence_transformers import SentenceTransformer


__all__ = ["vectors_get_embedding_minilm"]


def vectors_get_embedding_minilm(
    text: Union[List[str], str],
    bs: int = 32
) -> List[Union[List[float], float]]:
    """ Computes the MiniLM-V6 Hugginface vector embedding of the given input.

    Args:
        text (Union[List[str], str]): Text to be embedded. It can be a list of 
            strings, or a list of lists of strings.
        bs (int, optional): Batch size to be processes in parallel by the GPU.
            If not given, it is set to 32.

    Returns:
        List[Union[List[float], float]]: Returns the vector embedding being
            only a list of floats if the input was a string, or a list of lists
            of floats if the input was a list of lists of strings.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(text, device="cuda", batch_size=bs).tolist()


if __name__ == "__main__":
    test_embedding = vectors_get_embedding_minilm("TEST STRING")
    print(test_embedding)
