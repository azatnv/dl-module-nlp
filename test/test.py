from sentence_transformers import SentenceTransformer, CrossEncoder
from search_utils.search import (
    find_kNN_sentences_by_vectors,
    find_k_most_similar_sentences,
)
import pandas as pd

from pathlib import Path
from datetime import datatime
import argparse
import os

FILE = Path(__file__).resolve()
ROOT_ABS = FILE.parents[0]
ROOT = Path(os.path.relpath(ROOT_ABS, Path.cwd()))

EMBEDDER_PATH = f"{Path.cwd()}/models/embedding_model"
CLASSIFIER_PATH = f"{Path.cwd()}/models/classification_model"

top_k_vectors = 25
top_k_sentences = 15


def is_duplicates_pred(
    name_1,
    name_2,
    classificator,
    db_SentenceToVector,
    corpus_embeddings,
    corpus_sentences,
    top_k_vectors=top_k_vectors,
    top_k_sentences=top_k_sentences,
):
    # find (top_k_vectors) nearest sentences through cosine similarity of vectors
    name_1_vector = db_SentenceToVector[name_1]
    predicted_nearest_sentences = find_kNN_sentences_by_vectors(
        name_1_vector, corpus_embeddings, corpus_sentences, k=top_k_vectors
    )

    # find (top_k_setnences) most similar sentences using classification
    most_similar_sentences = find_k_most_similar_sentences(
        name_1, predicted_nearest_sentences, classificator, k=top_k_sentences
    )

    return name_2 in most_similar_sentences


def test(file=f"{ROOT}/companies_test.csv"):
    df = pd.read_csv(
        file,
        index_col="pair_id",
    )

    all_unique_companies = list(set(df[["name_1", "name_2"]].to_numpy().flatten()))[1:]

    embedder = SentenceTransformer(EMBEDDER_PATH)
    classificator = CrossEncoder(CLASSIFIER_PATH)

    corpus_sentences = list(all_unique_companies)
    corpus_embeddings = embedder.encode(corpus_sentences)
    db_SentenceToVector = {
        sentence: corpus_embeddings[i] for i, sentence in enumerate(corpus_sentences)
    }

    TP, FP, TN, FN = 0, 0, 0, 0
    for index_label, row in df.iterrows():
        name_1 = row["name_1"]
        name_2 = row["name_2"]
        is_duplicate_actual = True if row["is_duplicate"] == 1 else False

        if is_duplicates_pred(
            name_1,
            name_2,
            classificator,
            db_SentenceToVector,
            corpus_embeddings,
            corpus_sentences,
        ):
            if is_duplicate_actual:
                TP += 1
            else:
                FP += 1
        else:
            if is_duplicate_actual:
                FN += 1
            else:
                TN += 1

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)

    print(f"Precision: {Precision}\nRecall: {Recall}\nF1_score: {F1_score}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=f"{ROOT}/companies_test.csv",
        help="Test file *.csv",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    print(opt)
    test(**vars(opt))
