import numpy as np
import pandas as pd
import string
from collections import Counter, defaultdict
from math import log
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def create_term_document_matrix(play_text, vocab, play_names):
    vocab_index = {word: i for i, word in enumerate(vocab)}
    play_index = {play: i for i, play in enumerate(play_names)}
    matrix = np.zeros((len(vocab), len(play_names)))

    for _, row in play_text.iterrows():
        play = row['play_name']
        if play not in play_index:
            continue
        col = play_index[play]
        words = row['line'].lower().translate(str.maketrans('', '', string.punctuation)).split()
        for word in words:
            if word in vocab_index:
                row_index = vocab_index[word]
                matrix[row_index, col] += 1
    return matrix



def create_term_context_matrix(play_text, vocab, window_size=4):
    vocab_index = {word: i for i, word in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(vocab)))

    for _, row in play_text.iterrows():
        words = row['line'].lower().translate(str.maketrans('', '', string.punctuation)).split()
        for i, word in enumerate(words):
            if word not in vocab_index:
                continue
            row_idx = vocab_index[word]
            for j in range(max(i - window_size, 0), min(i + window_size + 1, len(words))):
                if i == j or words[j] not in vocab_index:
                    continue
                col_idx = vocab_index[words[j]]
                matrix[row_idx, col_idx] += 1
    return matrix


def create_PPMI_matrix(matrix):
    total = np.sum(matrix)
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)

    ppmi_matrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            p_ij = matrix[i][j] / total
            p_i = row_sums[i] / total
            p_j = col_sums[j] / total
            if p_ij > 0:
                ppmi_matrix[i][j] = max(log(p_ij / (p_i * p_j), 2), 0)
    return ppmi_matrix



def compute_tf_idf_matrix(matrix):
    tf = matrix
    df = np.count_nonzero(matrix, axis=1)
    idf = np.log(matrix.shape[1] / (df + 1))
    tf_idf = tf * idf[:, None]
    return tf_idf



def compute_cosine_similarity(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
 

def compute_jaccard_similarity(vec1, vec2):
    set1 = set(np.nonzero(vec1)[0])
    set2 = set(np.nonzero(vec2)[0])
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_dice_similarity(vec1, vec2):
    j = compute_jaccard_similarity(vec1, vec2)
    return (2 * j) / (j + 1) if j + 1 > 0 else 0.0


def rank_plays(matrix, play_names, target_index, similarity_fn):
    target_vec = matrix[:, target_index]
    similarities = []
    for i in range(matrix.shape[1]):
        if i == target_index:
            continue
        sim = similarity_fn(target_vec, matrix[:, i])
        similarities.append((play_names[i], sim))
    return sorted(similarities, key=lambda x: x[1], reverse=True)



def rank_words(matrix, vocab, target_index, similarity_fn):
    target_vec = matrix[target_index, :]
    similarities = []
    for i in range(matrix.shape[0]):
        if i == target_index:
            continue
        sim = similarity_fn(target_vec, matrix[i, :])
        similarities.append((vocab[i], sim))
    return sorted(similarities, key=lambda x: x[1], reverse=True)



def visualize_distances(matrix, labels, title="2D PCA görselleştirme"):

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(matrix.T) 

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        x, y = reduced[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, label, fontsize=12)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()