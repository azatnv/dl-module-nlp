from faiss import read_index
from sentence_transformers import SentenceTransformer, CrossEncoder
from search_utils.search import (
    find_kNN_sentences_by_vectors,
    find_k_most_similar_sentences,
)
import numpy as np

from datetime import datetime
from pathlib import Path
import pickle
import os

FILE = Path(__file__).resolve()
ROOT_ABS = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT_ABS, Path.cwd()))

EMBEDDER_PATH = f"{ROOT}/models/embedding_model"
CLASSIFICATOR_PATH = f"{ROOT}/models/classification_model"
faiss_index_path = f"{ROOT}/data/faiss-index"
embeddings_cache_path = f"{ROOT}/data/train-embeddings"

with open(embeddings_cache_path, "rb") as file:
    cached_data = pickle.load(file)
    corpus_sentences = cached_data["sentences"]
    corpus_embeddings = cached_data["embeddings"]

print("Loading models..")
embedder = SentenceTransformer(EMBEDDER_PATH)
classificator = CrossEncoder(CLASSIFICATOR_PATH)
index = read_index(faiss_index_path)
print("Done")

top_k_vectors = 25
top_k_sentences = 15
search_methods = ["knn", "ann"]
while True:
    method = input("\nSelect the search method (knn/ann): ")
    if method not in search_methods:
        continue

    query = input("\nPlease enter a company name: ")
    start_time = datetime.now()

    query_embedding = embedder.encode(query)

    if method == search_methods[0]:
        predicted_nearest_sentences = find_kNN_sentences_by_vectors(
            query_embedding, corpus_embeddings, corpus_sentences, k=top_k_vectors
        )
    else:
        norm_query_embedding = query_embedding / np.linalg.norm(query_embedding)
        norm_query_embedding = np.expand_dims(norm_query_embedding, axis=0)
        distances, corpus_ids = index.search(norm_query_embedding, top_k_vectors)
        predicted_nearest_sentences = [
            corpus_sentences[corpus_id] for corpus_id in corpus_ids[0]
        ]

    most_similar_sentences = find_k_most_similar_sentences(
        query, predicted_nearest_sentences, classificator, k=top_k_sentences
    )

    print(f"Input company: {query}")
    print(f"Similar companies (after {datetime.now() - start_time} seconds):")
    for i, response in enumerate(most_similar_sentences):
        print(f"\t{i + 1}. {response}")
