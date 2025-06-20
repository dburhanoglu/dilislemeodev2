import numpy as np
import pandas as pd
import string
from collections import defaultdict
from math import log
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_data():
    
    plays = pd.read_csv("data/will_play_text.csv", sep=";", 
                       names=['id','play','act','scene','speaker','line'], 
                       header=None)
    
    with open("data/vocab.txt") as f:
        vocab = [word.strip() for word in f]
    
    with open("data/play_names.txt") as f:
        play_names = [name.strip() for name in f]
    
    return plays, vocab, play_names

def create_term_doc_matrix(plays, vocab, play_names):
    word_idx = {word:i for i,word in enumerate(vocab)}
    play_idx = {play:i for i,play in enumerate(play_names)}
    
    matrix = np.zeros((len(vocab), len(play_names)))
    
    for _, row in plays.iterrows():
        play = row['play']
        if play not in play_idx:
            continue
            
        words = row['line'].lower().translate(
            str.maketrans('','',string.punctuation)).split()
        
        for word in words:
            if word in word_idx:
                matrix[word_idx[word], play_idx[play]] += 1
                
    return matrix

def create_term_context_matrix(plays, vocab, window=4):
    word_idx = {word:i for i,word in enumerate(vocab)}
    matrix = np.zeros((len(vocab), len(vocab)))
    
    for _, row in plays.iterrows():
        words = row['line'].lower().translate(
            str.maketrans('','',string.punctuation)).split()
        
        for i, word in enumerate(words):
            if word not in word_idx:
                continue
                
            start = max(0, i-window)
            end = min(len(words), i+window+1)
            
            for j in range(start, end):
                if i != j and words[j] in word_idx:
                    matrix[word_idx[word], word_idx[words[j]]] += 1
                    
    return matrix

def compute_ppmi(matrix):
    total = np.sum(matrix)
    row_sums = np.sum(matrix, axis=1)
    col_sums = np.sum(matrix, axis=0)
    
    ppmi = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i,j] > 0:
                p_ij = matrix[i,j]/total
                p_i = row_sums[i]/total
                p_j = col_sums[j]/total
                if p_i > 0 and p_j > 0:
                    ppmi[i,j] = max(log(p_ij/(p_i*p_j)), 0)
    return ppmi

def compute_tfidf(matrix):
    tf = matrix
    df = np.sum(matrix > 0, axis=1)
    idf = np.log(matrix.shape[1]/(df+1))
    return tf * idf[:,None]

def cosine_sim(vec1, vec2):
    norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
    return np.dot(vec1,vec2)/(norm1*norm2) if norm1*norm2 else 0

def visualize_pca(matrix, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(matrix.T)
    
    plt.figure(figsize=(10,8))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i,0], reduced[i,1])
        plt.text(reduced[i,0]+0.01, reduced[i,1]+0.01, label)
    plt.show()

def main():
    # Veri yükle
    plays, vocab, play_names = load_data()
    
    # 1. Terim-belge matrisi
    td_matrix = create_term_doc_matrix(plays, vocab, play_names)
    
    # 2. Oyun benzerlikleri
    visualize_pca(td_matrix, play_names)
    
    # 3. Kelime benzerlikleri
    tc_matrix = create_term_context_matrix(plays, vocab)
    
    # Örnek analizler
    target_play = "Hamlet"
    if target_play in play_names:
        play_idx = play_names.index(target_play)
        play_vec = td_matrix[:,play_idx]
        
        similarities = []
        for i in range(td_matrix.shape[1]):
            if i != play_idx:
                sim = cosine_sim(play_vec, td_matrix[:,i])
                similarities.append((play_names[i], sim))
        
        print(f"{target_play} en benzer 5 oyun:")
        for play, sim in sorted(similarities, key=lambda x: -x[1])[:5]:
            print(f"{play}: {sim:.3f}")
    
    target_word = "king"
    if target_word in vocab:
        word_idx = vocab.index(target_word)
        
        # Ham benzerlik
        word_vec = tc_matrix[word_idx,:]
        sims = [(vocab[i], cosine_sim(word_vec, tc_matrix[i,:])) 
               for i in range(len(vocab)) if i != word_idx]
        
        print(f"\n{target_word} en benzer 10 kelime:")
        for word, sim in sorted(sims, key=lambda x: -x[1])[:10]:
            print(f"{word}: {sim:.3f}")

if __name__ == "__main__":
    main()