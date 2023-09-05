import numpy as np


def calculate_cosine_similarity(frame_vector1: np.ndarray, frame_vector2: np.ndarray) -> float:
    cosine_similarity = np.linalg.norm(frame_vector1 - frame_vector2, 1) * 100 / len(frame_vector2)
    return cosine_similarity


def calculate_start_index(index: int, delta_t: int) -> int:
    if index < delta_t:
        return 0
    else:
        return index - delta_t


def calculate_end_index(index: int, delta_t: int, frame_length: int) -> int:
    if (index + delta_t) > frame_length:
        return frame_length
    else:
        return index + delta_t
