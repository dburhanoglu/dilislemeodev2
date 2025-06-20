import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from math import log
from sklearn.decomposition import PCA

# 1. VERİ ÖN İŞLEME FONKSİYONLARI
# ================================

def clean_play_name(name):
    """Oyun isimlerini standartlaştır"""
    if isinstance(name, str):
        # Tırnak işaretlerini kaldır
        name = re.sub(r'^"|"$', '', name)
        # Roma rakamlarını standartlaştır
        name = re.sub(r'Henry IV', 'Henry 4', name)
        name = re.sub(r'Henry V', 'Henry 5', name)
        name = re.sub(r'Richard III', 'Richard 3', name)
        # Fazla boşlukları kaldır
        name = re.sub(r'\s+', ' ', name).strip()
    return name

def clean_line_text(text):
    """Metin satırlarını temizle"""
    if isinstance(text, str):
        # Tırnak işaretlerini kaldır
        text = re.sub(r'^"|"$', '', text)
        # Noktalama işaretlerini kaldır
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Küçük harfe çevir
        return text.lower()
    return ""

# 2. VERİ YÜKLEME FONKSİYONU
# ==========================

def verileri_yukle():
    column_names = ['line_id', 'play_name', 'act', 'scene', 'speaker', 'line']
    
    # CSV'yi yüklerken dikkatli ol
    try:
        play_text = pd.read_csv(
            "data/will_play_text.csv", 
            sep=";", 
            names=column_names, 
            header=None,
            on_bad_lines='skip',
            encoding='latin1',
            quoting=3
        )
    except Exception as e:
        print(f"CSV yükleme hatası: {str(e)}")
        # Alternatif yaklaşım
        with open("data/will_play_text.csv", "r", encoding='latin1') as f:
            lines = f.readlines()
        data = [line.strip().split(";", maxsplit=5) for line in lines]
        play_text = pd.DataFrame(data, columns=column_names)
    
    # Oyun isimlerini temizle
    play_text['play_name'] = play_text['play_name'].apply(clean_play_name)
    play_text['line'] = play_text['line'].apply(clean_line_text)
    
    # Kelime haznesini yükle
    try:
        with open("data/vocab.txt", "r", encoding='utf-8') as f:
            sozluk = [line.strip().lower() for line in f]
    except:
        with open("data/vocab.txt", "r", encoding='latin1') as f:
            sozluk = [line.strip().lower() for line in f]
    
    # Oyun isimlerini yükle ve temizle
    try:
        with open("data/play_names.txt", "r", encoding='utf-8') as f:
            oyun_isimleri = [clean_play_name(line.strip()) for line in f]
    except:
        with open("data/play_names.txt", "r", encoding='latin1') as f:
            oyun_isimleri = [clean_play_name(line.strip()) for line in f]
    
    # Benzersiz oyun isimlerini kontrol et
    unique_plays = play_text['play_name'].unique()
    print("\nCSV'deki benzersiz oyunlar:")
    for play in unique_plays:
        print(f"- {play}")
    
    print("\nplay_names.txt'deki oyunlar:")
    for play in oyun_isimleri:
        print(f"- {play}")
    
    # Eşleşmeyen oyunları bul
    unmatched = set(oyun_isimleri) - set(unique_plays)
    if unmatched:
        print(f"\nUYARI: Eşleşmeyen oyunlar: {unmatched}")
        # Eşleşmeyen oyunları oyun isimleri listesinden çıkar
        oyun_isimleri = [play for play in oyun_isimleri if play not in unmatched]
    
    return play_text, sozluk, oyun_isimleri

# 3. MATRIS OLUŞTURMA FONKSIYONLARI
# =================================

def terim_belge_matrisi_olustur(play_text, sozluk, oyun_isimleri):
    sozluk_indeksi = {kelime: i for i, kelime in enumerate(sozluk)}
    oyun_indeksi = {oyun: i for i, oyun in enumerate(oyun_isimleri)}
    matris = np.zeros((len(sozluk), len(oyun_isimleri)))
    
    # Hata ayıklama için sayaç
    total_words = 0
    matched_plays = set()
    
    for _, row in play_text.iterrows():
        oyun = row['play_name']
        if oyun not in oyun_indeksi:
            continue
            
        sutun = oyun_indeksi[oyun]
        matched_plays.add(oyun)
        kelimeler = row['line'].split()
        
        for kelime in kelimeler:
            if kelime in sozluk_indeksi:
                satir_indeksi = sozluk_indeksi[kelime]
                matris[satir_indeksi, sutun] += 1
                total_words += 1
    
    print(f"\nToplam işlenen kelime: {total_words}")
    print(f"Eşleşen oyunlar: {matched_plays}")
    
    # Sıfır vektör kontrolü
    zero_plays = []
    for i, play in enumerate(oyun_isimleri):
        if np.sum(matris[:, i]) == 0:
            print(f"UYARI: '{play}' için tüm vektör sıfır!")
            zero_plays.append(i)
    
    # Sıfır vektörleri kaldır
    if zero_plays:
        print(f"Sıfır vektörlü oyunlar kaldırılıyor: {[oyun_isimleri[i] for i in zero_plays]}")
        non_zero_indices = [i for i in range(len(oyun_isimleri)) if i not in zero_plays]
        matris = matris[:, non_zero_indices]
        oyun_isimleri = [oyun_isimleri[i] for i in non_zero_indices]
    
    return matris, oyun_isimleri

def terim_icerik_matrisi_olustur(play_text, sozluk, pencere_buyuklugu=4):
    sozluk_indeksi = {kelime: i for i, kelime in enumerate(sozluk)}
    matris = np.zeros((len(sozluk), len(sozluk)))
    
    total_pairs = 0
    
    for _, row in play_text.iterrows():
        kelimeler = row['line'].split()
        for i, kelime in enumerate(kelimeler):
            if kelime not in sozluk_indeksi:
                continue
            satir_indeksi = sozluk_indeksi[kelime]
            start = max(i - pencere_buyuklugu, 0)
            end = min(i + pencere_buyuklugu + 1, len(kelimeler))
            
            for j in range(start, end):
                if i == j or kelimeler[j] not in sozluk_indeksi:
                    continue
                sutun_indeksi = sozluk_indeksi[kelimeler[j]]
                matris[satir_indeksi, sutun_indeksi] += 1
                total_pairs += 1
    
    print(f"\nToplam kelime çifti: {total_pairs}")
    return matris

def PPMI_matrisi_olustur(matris):
    total = np.sum(matris)
    if total == 0:
        return matris
    
    row_sums = np.sum(matris, axis=1)
    col_sums = np.sum(matris, axis=0)
    
    ppmi_matrix = np.zeros_like(matris)
    
    for i in range(matris.shape[0]):
        for j in range(matris.shape[1]):
            if matris[i, j] > 0:
                p_ij = matris[i, j] / total
                p_i = row_sums[i] / total
                p_j = col_sums[j] / total
                
                if p_i > 0 and p_j > 0:
                    pmi = np.log(p_ij / (p_i * p_j))
                    ppmi_matrix[i, j] = max(pmi, 0)
    
    return ppmi_matrix

def TF_IDF_matrisi_hesapla(matris):
    # Term Frequency (TF)
    tf = matris
    
    # Document Frequency (DF)
    df = np.sum(matris > 0, axis=1)
    
    # Inverse Document Frequency (IDF)
    n_docs = matris.shape[1]
    idf = np.log(n_docs / (df + 1)) + 1  # Smoothing
    
    # TF-IDF
    tf_idf = tf * idf[:, np.newaxis]
    
    return tf_idf

# 4. BENZERLİK FONKSİYONLARI
# ==========================

def kosinus_benzerligi_hesapla(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def jaccard_benzerligi_hesapla(vec1, vec2):
    set1 = set(np.nonzero(vec1)[0])
    set2 = set(np.nonzero(vec2)[0])
    
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    if union == 0:
        return 0.0
    return intersection / union

def dice_benzerligi_hesapla(vec1, vec2):
    j = jaccard_benzerligi_hesapla(vec1, vec2)
    if j == 0:
        return 0.0
    return (2 * j) / (j + 1)

# 5. SIRALAMA FONKSİYONLARI
# =========================

def oyunlari_sirala(matris, oyun_isimleri, hedef_indeks, benzerlik_fonksiyonu):
    hedef_vektor = matris[:, hedef_indeks]
    similarities = []
    
    for i in range(matris.shape[1]):
        if i == hedef_indeks:
            continue
        sim = benzerlik_fonksiyonu(hedef_vektor, matris[:, i])
        similarities.append((oyun_isimleri[i], sim))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)

def kelimeleri_sirala(matris, sozluk, hedef_indeks, benzerlik_fonksiyonu):
    hedef_vektor = matris[hedef_indeks, :]
    similarities = []
    
    for i in range(matris.shape[0]):
        if i == hedef_indeks:
            continue
        sim = benzerlik_fonksiyonu(hedef_vektor, matris[i, :])
        similarities.append((sozluk[i], sim))
    
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# 6. GÖRSELLEŞTİRME FONKSİYONU
# ============================

def mesafeleri_gorselleştir(matris, etiketler, baslik="2D PCA görselleştirme"):
    # Sıfır olmayan vektörleri filtrele
    non_zero_indices = [i for i in range(matris.shape[1]) if np.any(matris[:, i])]
    if not non_zero_indices:
        print("Görselleştirme yapılamadı: tüm vektörler sıfır!")
        return
    
    filtered_matrix = matris[:, non_zero_indices]
    filtered_labels = [etiketler[i] for i in non_zero_indices]
    
    # PCA uygula
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(filtered_matrix.T)
    
    # Görselleştir
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(filtered_labels):
        x, y = reduced[i]
        plt.scatter(x, y, color='blue')
        plt.text(x + 0.01, y + 0.01, label, fontsize=9, alpha=0.8)
    
    plt.title(baslik)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('oyun_benzerlikleri.png', dpi=300)
    plt.show()

# 7. ANA FONKSİYON
# ================

def main():
    try:
        print("Veriler yükleniyor...")
        play_text, sozluk, oyun_isimleri = verileri_yukle()
        
        print("\nTerim-belge matrisi oluşturuluyor...")
        td_matris, oyun_isimleri = terim_belge_matrisi_olustur(play_text, sozluk, oyun_isimleri)
        
        print(f"\nMatris boyutu: {td_matris.shape}")
        print(f"Sıfır olmayan öğe sayısı: {np.count_nonzero(td_matris)}")
        
        print("\nOyun benzerlikleri görselleştiriliyor...")
        mesafeleri_gorselleştir(td_matris, oyun_isimleri, "Shakespeare Oyunlarının Benzerlikleri")
        
        # Oyun benzerlik analizi
        target_plays = ["King Lear", "Hamlet", "Macbeth", "Othello", "A Midsummer Night's Dream"]
        
        for target_play in target_plays:
            if target_play in oyun_isimleri:
                target_idx = oyun_isimleri.index(target_play)
                
                if np.any(td_matris[:, target_idx]):
                    print(f"\n{target_play} için en benzer 5 oyun (kosinüs benzerliği):")
                    similar_plays = oyunlari_sirala(td_matris, oyun_isimleri, target_idx, kosinus_benzerligi_hesapla)
                    for play, sim in similar_plays[:5]:
                        print(f"  {play}: {sim:.4f}")
                else:
                    print(f"\n{target_play} için vektör sıfır, benzerlik hesaplanamıyor")
            else:
                print(f"\n{target_play} oyun isimleri listesinde bulunamadı")
        
        print("\nTerim-bağlam matrisi oluşturuluyor...")
        tc_matris = terim_icerik_matrisi_olustur(play_text, sozluk)
        
        # Kelime benzerlik analizi
        target_words = ["king", "love", "death", "fool", "heart"]
        
        for target_word in target_words:
            if target_word in sozluk:
                target_idx = sozluk.index(target_word)
                
                print(f"\n{target_word} için en benzer 10 kelime:")
                
                # Ham benzerlik
                similar_words = kelimeleri_sirala(tc_matris, sozluk, target_idx, kosinus_benzerligi_hesapla)
                print(f"  Ham matris:")
                for word, sim in similar_words[:10]:
                    print(f"    {word}: {sim:.4f}")
                
                # PPMI
                ppmi_matrix = PPMI_matrisi_olustur(tc_matris)
                similar_words_ppmi = kelimeleri_sirala(ppmi_matrix, sozluk, target_idx, kosinus_benzerligi_hesapla)
                print(f"  PPMI matrisi:")
                for word, sim in similar_words_ppmi[:10]:
                    print(f"    {word}: {sim:.4f}")
                
                # TF-IDF
                tfidf_matrix = TF_IDF_matrisi_hesapla(tc_matris)
                similar_words_tfidf = kelimeleri_sirala(tfidf_matrix, sozluk, target_idx, kosinus_benzerligi_hesapla)
                print(f"  TF-IDF matrisi:")
                for word, sim in similar_words_tfidf[:10]:
                    print(f"    {word}: {sim:.4f}")
            else:
                print(f"\n{target_word} kelimesi sözlükte bulunamadı")
    
    except Exception as e:
        print(f"\nKRİTİK HATA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()