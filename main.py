import pandas as pd
from vsm import create_term_document_matrix, visualize_distances
import matplotlib.pyplot as plt


column_names = ['line_id', 'play_name', 'act', 'scene', 'speaker', 'line']
play_text = pd.read_csv("data/will_play_text.csv", sep=";", names=column_names, header=None)


vocab = [line.strip() for line in open("data/vocab.txt")]
play_names = [line.strip() for line in open("data/play_names.txt")]


td_matrix = create_term_document_matrix(play_text, vocab, play_names)


visualize_distances(td_matrix, play_names)
