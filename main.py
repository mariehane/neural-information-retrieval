import sys
import time
import json
import zipfile
import argparse
import xmltodict
import pytrec_eval
import spacy
import tqdm
import optuna
import torch
import numpy as np
import pandas as pd
import pyterrier as pt
from datetime import datetime
from pprint import pprint
from pathlib import Path
from sklearn.model_selection import train_test_split

if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

from data import Cord19Dataset, convert_qrels_to_pyterrier_format, convert_topics_to_pyterrier_format
from preprocess import remove_stopwords, stem_porter, stem_snowball, lemmatize_wordnet, lemmatize_lemminflect, remove_duplicates, remove_without_title, remove_before
from index import TerrierIndex
from models import create_models, tune_bm25, GensimQueryExpander, DistilBertLM
from evaluate import output_run

def main():
    parser = argparse.ArgumentParser(description="one of '--compare-indexes', '--train-validate', or '--produce-eval-runs' must be enabled.")
    parser.add_argument("--compare-indexes", action='store_true', help="compare various preprocessing/indexing methods")
    parser.add_argument("--train-validate", action='store_true', help="train and validate various information retrieval models")
    parser.add_argument("--produce-eval-runs", action='store_true', help="create final evaluation runs")
    parser.add_argument("--data_dir", type=str, default='data', help="directory to load/store data in")
    parser.add_argument("--runs_dir", type=str, default='runs', help="directory to store output runs in")
    #parser.add_argument("--indexes_dir", type=str, default='indexes', help="directory for all PyTerrier indexes")
    parser.add_argument("--n_papers", type=int, default=None, help="how many papers to use - if not specified then all are used")
    parser.add_argument("--seed", type=int, default=None, help="random seed - specify in order to have reproducable runs")
    config = parser.parse_args()
    
    if not config.compare_indexes and not config.train_validate and not config.produce_eval_runs:
        parser.print_help()
        sys.exit(0)
    
    path_data = Path(config.data_dir)
    path_df = path_data / "df.csv"

    if config.seed is not None:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        #torch.use_deterministic_algorithms(True)

    print("> Loading dataset")
    dataset = Cord19Dataset(base_dir=str(path_data))
    if config.n_papers is None:
        config.n_papers = len(dataset.metadata)

    loaded_dataframe = False
    if path_df.exists():
        print("> Loading dataframe...")
        df = pd.read_csv(str(path_df))
        loaded_dataframe = True
        if len(df) > config.n_papers:
            print(f"- dataframe has more papers than specified ({len(df)}), truncating...")
            df = df[:config.n_papers]
        elif len(df) < config.n_papers:
            print(f"- cached dataframe has less papers than specified ({len(df)}), need to recreate dataframe")
            loaded_dataframe = False

    if not loaded_dataframe:
        print("> Creating dataframe...")
        df = dataset.get_dataframe(config.n_papers, show_progressbar=True)
        df.to_csv(path_df)
    print(df)

    if config.compare_indexes:
        print("> Preprocessing with NLTK")
        print("- Removing stopwords from abstracts")
        start = time.time()
        df["abstract_nostopwords"] = remove_stopwords(df["abstract"], show_progressbar=True)
        print("- Took", round(time.time() - start, 2), "s")

        print("- Removing stopwords from texts")
        start = time.time()
        df["text_nostopwords"] = remove_stopwords(df["text"], show_progressbar=True)
        print("- Took", round(time.time() - start, 2), "s")

        print("- Using Porter Stemmer on abstracts")
        start = time.time()
        df["abstract_porter"] = stem_porter(df["abstract_nostopwords"], show_progressbar=True)
        print("- Took", round(time.time() - start, 2), "s")

        print("- Using Porter Stemmer on full texts, no stopword removal")
        start = time.time()
        df["text_porter_stopwords"] = stem_porter(df["text"], show_progressbar=True)
        print("- Took", round(time.time() - start, 2), "s")

        print("- Using Porter Stemmer on full texts")
        start = time.time()
        df["text_porter"] = stem_porter(df["text_nostopwords"], show_progressbar=True)
        print("- Took", round(time.time() - start, 2), "s")

        print("- Using Snowball Stemmer")
        start = time.time()
        df["text_snowball"] = stem_porter(df["text_nostopwords"], show_progressbar=True)
        print("- Took", round(time.time() - start, 2), "s")

        print("- Using WordNet Lemmatizer")
        start = time.time()
        df["text_wordnet"] = lemmatize_wordnet(df["text_nostopwords"], show_progressbar=True)
        print("- Took", round(time.time() - start, 2), "s")

        print("> Preprocessing with lemminflect")
        start = time.time()
        df["text_lemminflect"] = lemmatize_lemminflect(df["text_nostopwords"], show_progressbar=True)
        print("- Took", round(time.time() - start, 2), "s")
    print(df)

    print("> Preprocessing...")
    print("- Removing duplicates")
    df = remove_duplicates(df)
    #print("- Removing articles without title")
    #df = remove_without_title(df)
    #print("- Removing articles before 2000-01-01")
    #df = remove_before(df, date=datetime(2000, 1, 1))

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
    ]
    if config.compare_indexes:
        indexes += [
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
            TerrierIndex("Abstracts, no stopword removal/stemming", 
                        path="indexes/abstract_stopword",
                        text=df["abstract"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        store_positions=False,
                        remove_stopwords=True,
                        stem=None),
            TerrierIndex("Abstracts, store positions, no stopword removal/stemming", 
                        path="indexes/positions_abstract_stopword",
                        text=df["abstract"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        store_positions=True,
                        remove_stopwords=True,
                        stem=None),
            TerrierIndex("Texts, no stopword removal/stemming", 
                        path="indexes/text_stopword",
                        text=df["text"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        store_positions=False,
                        remove_stopwords=False,
                        stem=None),
            TerrierIndex("Texts, stopword removal, no stemming", 
                        path="indexes/text_stopword",
                        text=df["text"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        store_positions=False,
                        remove_stopwords=True,
                        stem=None),
            TerrierIndex("Texts, store positions, no stopword removal/stemming", 
                        path="indexes/positions_text_stopword",
                        text=df["text"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        store_positions=True,
                        remove_stopwords=False,
                        stem=None),
            TerrierIndex("Porter stemmer on abstracts", 
                        path="indexes/porter_abstract",
                        text=df["abstract_porter"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        remove_stopwords=False,
                        stem=None),
            TerrierIndex("Porter stemmer on full texts", 
                        path="indexes/porter_text",
                        text=df["text_porter"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        remove_stopwords=False,
                        stem=None),
            TerrierIndex("Snowball stemmer on full texts", 
                        path="indexes/snowball_text",
                        text=df["text_snowball"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        remove_stopwords=False,
                        stem=None),
            TerrierIndex("WordNet lemmatizer on full texts", 
                        path="indexes/wordnet_text",
                        text=df["text_wordnet"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        remove_stopwords=False,
                        stem=None),
            TerrierIndex("Lemminflect on full texts", 
                        path="indexes/lemminflect_text",
                        text=df["text_lemminflect"],
                        docno=df["cord_uid"],
                        metadata=[df["title"]],
                        remove_stopwords=False,
                        stem=None),
        ]

    print("> Indexing...")
    for index in indexes:
        if index.exists() and not config.compare_indexes:
            print("- Loading index:", index.name)
            index.load()
        else:
            print("- Creating index:", index.name)
            print("- Directory:", index.path)
            time_to_index = index.create()
            print("- Time to index:", time_to_index, "s")

        n_docs, n_unique_terms, n_tokens, index_size_mb = index.get_stats()
        print("- No. of docs indexed:", n_docs)
        print("- No. of unique terms:", n_unique_terms)
        print("- Total no. of terms:", n_tokens)
        print("- Index size: ", index_size_mb, "MB")
        print()

    # 2. Ranking models
    topics_query = convert_topics_to_pyterrier_format(dataset.topics_train, query_column="query")
    topics_question = convert_topics_to_pyterrier_format(dataset.topics_train, query_column="question")

    qrels = convert_qrels_to_pyterrier_format(dataset.qrels_train)
    print("> Dropping qrels that are not in the dataset")
    qrels = qrels.merge(df["cord_uid"], left_on="docno", right_on="cord_uid")
    print("- total qrels: ", len(dataset.qrels_train))
    print("- qrels in dataset: ", len(qrels))

    if config.compare_indexes or config.train_validate:
        print("> Splitting qrels into train/validation set")
        # TODO: Should I split by the topics instead?
        qrels_train, qrels_valid = train_test_split(qrels, random_state=config.seed)
        print("- train size:", len(qrels_train), "validation size:", len(qrels_valid))

    if config.compare_indexes:
        for i, index in enumerate(indexes):
            topics_compare_idx = topics_query.copy()
            if i == 9:
                topics_compare_idx["query"] = stem_porter(topics_compare_idx["query"])
            elif i == 10:
                topics_compare_idx["query"] = stem_snowball(topics_compare_idx["query"])
            elif i == 11:
                topics_compare_idx["query"] = lemmatize_wordnet(topics_compare_idx["query"])
            elif i == 12:
                topics_compare_idx["query"] = lemmatize_lemminflect(topics_compare_idx["query"])
            print("Running simple BM25 on index:", index.name)
            model = pt.BatchRetrieve(index.index, wmodel="BM25")
            results = pt.Experiment(
                retr_systems=[model],
                names=["BM25"],
                topics=topics_compare_idx,
                qrels=qrels_train,
                eval_metrics=["map", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "mrt"])
            print(results)

    if config.train_validate:
        # TODO: Try some rank fusion approaches to combined different retrieval model results.
        #       - CombMNZ
        #       - BordaCount
        #       - etc.

        # 3. Advanced Topics in Information Retrieval
        # TODO: Use word embeddings to do query expansion as done by Kuzi et al. 
        # TODO: Look at recent approaches proposed for the TREC-COVID track and evaluate their approaches (no need to reimplement/retrain models, just evaluate them) 
        # TODO: Tune and run at least 1 learning-to-rank approach
        #       - RankNet
        #       - LambdaMART
        #       - etc.

        # 4. Evaluation
        # TODO: use trec-eval: https://github.com/usnistgov/trec_eval
        # TODO: possibly report more metrics
        
        models = create_models(indexes[0], indexes[1], topics_query, qrels_train, seed=config.seed, verbose=True)

        print("> Evaluating systems on train qrels")
        results = pt.Experiment(
            retr_systems=models.values(),
            names=models.keys(),
            topics=topics_query,
            qrels=qrels_train,
            eval_metrics=["map", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "mrt"])
        print("Train results:")
        print(results)

        print("> Evaluating systems on validation qrels")
        results = pt.Experiment(
            retr_systems=models.values(),
            names=models.keys(),
            topics=topics_query,
            qrels=qrels_valid,
            eval_metrics=["map", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "mrt"])

        print("Validation results:")
        print(results)

    # 4.1 Real-World Use Case
    if config.produce_eval_runs:
        topics_query = convert_topics_to_pyterrier_format(dataset.topics_test, query_column="query")
        topics_question = convert_topics_to_pyterrier_format(dataset.topics_test, query_column="question")
        topics_question["query"] = topics_question["query"].str.replace("?","")

        print("> Retraining on full data")
        models = create_models(indexes[0], indexes[1], topics_query, qrels, seed=config.seed, verbose=True)

        print("> Generating final run files...")
        # output top 1000 (at most) documents
        for model_name in models:
            model = models[model_name]
            model_id = model_name.lower().replace(' ', '_')
            path_query = Path(config.runs_dir, f"dvf159.{model_id}.query")
            path_question = Path(config.runs_dir, f"dvf159.{model_id}.question")
            print(f"Outputting model '{model_name}' to {path_query} and {path_question}")
            output_run(model, model_id, topics_query, 'dvf159', str(path_query), n_docs_per_topic=1000, filter_out=dataset.qrels_train.cord_uid)
            output_run(model, model_id, topics_question, 'dvf159', str(path_question), n_docs_per_topic=1000, filter_out=dataset.qrels_train.cord_uid)

if __name__=="__main__":
    main()