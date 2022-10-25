from sentence_transformers import util
import numpy as np


def find_kNN_sentences_by_vectors(
    vector_query, corpus_embeddings, corpus_sentences, k=20
):
    hits = util.semantic_search(vector_query, corpus_embeddings, top_k=k)[0]

    return [corpus_sentences[hit["corpus_id"]] for hit in hits]


def find_k_most_similar_sentences(sentence_query, sentence_subset, model, k=5):
    sentences_combinations = [[sentence_query, s] for s in sentence_subset]

    similarity_scores = model.predict(sentences_combinations)
    sorted_indexes = list(reversed(np.argsort(similarity_scores)))

    return [sentence_subset[index_max_score] for index_max_score in sorted_indexes[:k]]
