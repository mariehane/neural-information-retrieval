import sys
import time
import json
import zipfile
import xmltodict
import pytrec_eval
import spacy
import tqdm
import optuna
import numpy as np
import pandas as pd
import pyterrier as pt
from pprint import pprint
from pathlib import Path
from sklearn.model_selection import train_test_split

from data import Cord19Dataset, convert_qrels_to_pyterrier_format, convert_topics_to_pyterrier_format
from preprocess import lemmatize_wordnet, stem_porter, stem_snowball, lemmatize_lemminflect
from index import TerrierIndex

# parameters:
n_papers = 5_000 #len(dataset.metadata)


print("> Loading dataset")
dataset = Cord19Dataset(base_dir='data', download=True)

print("> Creating simple dataframe...")
df = dataset.get_dataframe(n_papers, show_progressbar=True)
print(df)

print("> Preprocessing with NLTK lemmatizers")

print("- Using WordNet Stemmer")
start = time.time()
df["abstract_wordnet"] = lemmatize_wordnet(df["abstract"], show_progressbar=True)
print("- Took", round(time.time() - start, 2), "s")

print("- Using Porter Stemmer")
start = time.time()
df["abstract_porter"] = stem_porter(df["abstract"], show_progressbar=True)
print("- Took", round(time.time() - start, 2), "s")

print("- Using Snowball Stemmer")
start = time.time()
df["abstract_snowball"] = stem_porter(df["abstract"], show_progressbar=True)
print("- Took", round(time.time() - start, 2), "s")

#print("> Preprocessing with lemminflect")
#start = time.time()
#df["abstract_lemminflect"] = lemmatize_lemminflect(df["abstract"], show_progressbar=True)
#print("- Took", round(time.time() - start, 2), "s")

indexes = [
    TerrierIndex("Default, abstracts", 
                 path="indexes/default_abstract",
                 text=df["abstract"],
                 docno=df["cord_uid"],
                 metadata=[df["title"]]),
    TerrierIndex("Default, full paper text", 
                 path="indexes/default_text",
                 text=df["text"],
                 docno=df["cord_uid"],
                 metadata=[df["title"]]),
    TerrierIndex("Abstracts, store positions", 
                 path="indexes/positions_abstract",
                 text=df["abstract"],
                 docno=df["cord_uid"],
                 metadata=[df["title"]],
                 store_positions=True),
    TerrierIndex("Full texts, store positions", 
                 path="indexes/positions_text",
                 text=df["text"],
                 docno=df["cord_uid"],
                 metadata=[df["title"]],
                 store_positions=True),
    TerrierIndex("Porter stemmer on abstracts", 
                 path="indexes/porter_abstract",
                 text=df["abstract_porter"],
                 docno=df["cord_uid"],
                 metadata=[df["title"]],
                 store_positions=True),
]

print("> Indexing...")

for index in indexes:
    print("- Creating index:", index.name)
    print("- Directory:", index.path)

    time_to_index = index.create()
    n_docs, n_unique_terms, n_tokens, index_size_mb = index.get_stats()

    print("- Time to index:", time_to_index, "s")
    print("- No. of docs indexed:", n_docs)
    print("- No. of unique terms:", n_unique_terms)
    print("- Total no. of terms:", n_tokens)
    print("- Index size: ", index_size_mb, "MB")
    print()

#sys.exit(0)

# 2. Ranking models
topics = convert_topics_to_pyterrier_format(dataset.topics_train, query_column="query")

qrels = convert_qrels_to_pyterrier_format(dataset.qrels_train)
print("> Dropping qrels that are not in the dataset")
qrels = qrels.merge(df["cord_uid"], left_on="docno", right_on="cord_uid")
print("- total qrels: ", len(dataset.qrels_train))
print("- qrels in dataset: ", len(qrels))

print("> Splitting qrels into train/validation set")
qrels_train, qrels_valid = train_test_split(qrels)
print("- train size:", len(qrels_train), "validation size:", len(qrels_valid))

# TODO: Once you find the best configuration, train on the full data
# TODO: Tune and run BM25
# TODO: Tune and run a language model, that ranks docs based on the probability of the model generating the query
# TODO: Try some rank fusion approaches to combined different retrieval model results.
#       - CombSum
#       - CombMNZ
#       - BordaCount
#       - etc.
# TODO: Expand with even more complex ideas of your own.


# 3. Advanced Topics in Information Retrieval
# TODO: Use word embeddings to do query expansion as done by Kuzi et al. 
# TODO: Use BM25 or something similar to generate an initial ranking, and then re-rank the top K documents using contextual embeddings. 
# TODO: Look at recent approaches proposed for the TREC-COVID track and evaluate their approaches (no need to reimplement/retrain models, just evaluate them) 
# TODO: Tune and run at least 1 learning-to-rank approach
#       - RankNet
#       - LambdaMART
#       - etc.

index = indexes[1]

def objective(trial):
    c = trial.suggest_uniform('c', 0, 10)
    k_1 = trial.suggest_uniform('k_1', 0, 10)
    k_3 = trial.suggest_uniform('k_3', 0, 10)

    bm25 = pt.BatchRetrieve(index.index, wmodel="BM25", controls={"c": c, "bm25.k_1": k_1, "bm25.k_3": k_3})
    results = pt.Experiment(
        retr_systems=[bm25],
        names=['BM25'],
        topics=topics,
        qrels=qrels_valid,
        eval_metrics=["map", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20"])

    return results.loc[0, "map"]

print()
print(f"> Running optuna on index <{index.name}>")
study = optuna.create_study(study_name="BM25 Tuning", direction='maximize')
study.optimize(objective, n_trials=100)

print("- Best params:")
print(study.best_params)

# 4. Evaluation
# TODO: use trec-eval: https://github.com/usnistgov/trec_eval
# TODO: Report MAP and NDCG at cut-offs of 5, 10, 20.
# TODO: possibly report more metrics
# TODO: report mean response times of your systems
# 4.1 Real-World Use Case
# TODO: Output submissions in the TREC run format

# TODO: FILTER OUT DOCUMENTS FROM qrels_train!

for index in indexes:
    print("> Evaluating models on all indexes")
    print("- Computing results for", index.name)
    tf = pt.BatchRetrieve(index.index, wmodel="Tf")
    bm25v1 = pt.BatchRetrieve(index.index, wmodel="BM25")  # default parameters
    bm25v2 = pt.BatchRetrieve(index.index, wmodel="BM25", controls={"c": 0.1, "bm25.k_1": 2.0, "bm25.k_3": 10})
    bm25v3 = pt.BatchRetrieve(index.index, wmodel="BM25", controls={"c": 8, "bm25.k_1": 1.4, "bm25.k_3": 10})

    bm25best = pt.BatchRetrieve(index.index, wmodel="BM25", controls={"c": study.best_params["c"], 
                                                                    "bm25.k_1": study.best_params["k_1"],
                                                                    "bm25.k_3": study.best_params["k_3"]})

    results = pt.Experiment(
        retr_systems=[tf, bm25v1, bm25v2, bm25v3, bm25best],
        names=['TF', 'BM25v1', 'BM25v2', 'BM25v3', 'BM25 (grid search)'],
        topics=topics,
        qrels=qrels_valid,
        eval_metrics=["map", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20"])

    print(results)


#sys.exit(0) # only doing evaluation at the veery end
