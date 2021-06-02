import sys
import time
import json
import zipfile
import xmltodict
import pytrec_eval
import spacy
import tqdm
import numpy as np
import pandas as pd
import pyterrier as pt
from pprint import pprint
from pathlib import Path

from data import Cord19Dataset
from preprocess import lemmatize_wordnet, stem_porter, stem_snowball, lemmatize_lemminflect
from index import TerrierIndex

print("> Loading dataset")
dataset = Cord19Dataset(base_dir='data', download=True)
n_papers = 5_000 #len(dataset.metadata)

print("> Creating simple dataframe...")
df = dataset.get_dataframe(n_papers, show_progressbar=True)
print(df)

print("> Preprocessing with NLTK lemmatizers")

print("- Using WordNet Stemmer")
start = time.time()
df["abstract_wordnet"] = lemmatize_wordnet(df["abstract"], show_progressbar=True)
print(df["abstract_wordnet"])
print("- Took", round(time.time() - start, 2), "s")

print("- Using Porter Stemmer")
start = time.time()
df["abstract_porter"] = stem_porter(df["abstract"], show_progressbar=True)
print(df["abstract_porter"])
print("- Took", round(time.time() - start, 2), "s")

print("- Using Snowball Stemmer")
start = time.time()
df["abstract_snowball"] = stem_porter(df["abstract"], show_progressbar=True)
print(df["abstract_snowball"])
print("- Took", round(time.time() - start, 2), "s")

print("> Preprocessing with lemminflect")
start = time.time()
df["abstract_lemminflect"] = lemmatize_lemminflect(df["abstract"], show_progressbar=True)
print(df["abstract_lemminflect"])
print("- Took", round(time.time() - start, 2), "s")

# TODO: Index the dataset with Terrier, Indri, Elasticsearch, etc.
# TODO: Try diff. approaches to optimize index and see impact of each approach
#       - stemming
#       - lemmatization
indexes = [
    {
        "name": "Default, abstracts",
        "folder_name": "default_abstract",
        "text": df["abstract"],
        "metadata": [
            df["cord_uid"],
            df["title"]
            #df["publish_time"],
            #df["journal"])
        ],
        "stopwords_removal": True,
        "tokeniser": "EnglishTokeniser", # UTFTokenizer
        "store_positions": False
    },
    {
        "name": "Default, full paper text",
        "folder_name": "default_text",
        "text": df["text"],
        "metadata": [
            df["cord_uid"],
            df["title"]
        ],
        "stopwords_removal": True,
        "tokeniser": "EnglishTokeniser",
        "store_positions": False
    },
    {
        "name": "Abstracts, store positions",
        "folder_name": "positions_abstract",
        "text": df["abstract"],
        "metadata": [
            df["cord_uid"],
            df["title"]
        ],
        "stopwords_removal": True,
        "tokeniser": "EnglishTokeniser",
        "store_positions": True
    },
    {
        "name": "Full texts, store positions",
        "folder_name": "positions_text",
        "text": df["text"],
        "metadata": [
            df["cord_uid"],
            df["title"]
        ],
        "stopwords_removal": True,
        "tokeniser": "EnglishTokeniser",
        "store_positions": True
    },
    {
        "name": "Lemminflect on abstracts",
        "folder_name": "lemminflect_abstract",
        "text": df["abstract_lemminflect"],
        "metadata": [
            df["cord_uid"],
            df["title"]
            #df["publish_time"],
            #df["journal"])
        ],
        "stopwords_removal": True,
        "tokeniser": "EnglishTokeniser", # UTFTokenizer
        "store_positions": False
    },
]

print("> Indexing...")
if not pt.started():
    pt.init()

for index_dict in indexes:
    print("- Creating index:", index_dict["name"])
    index_path = str(Path("./indexes") / index_dict["folder_name"])
    print("- Directory:", index_path)

    indexer = pt.DFIndexer("./" + index_path, overwrite=True, blocks=index_dict["store_positions"])
    if not index_dict["stopwords_removal"]:
        indexer.setProperty("termpipelines", "")
    if index_dict["tokeniser"] != "EnglishTokeniser":
        indexer.setProperty("tokeniser", index_dict["tokeniser"])

    start = time.time()
    index_ref = indexer.index(index_dict["text"], *index_dict["metadata"])
    end = time.time()

    index = pt.IndexFactory.of(index_ref)
    stats = index.getCollectionStatistics()
    print("- Time to index:", round(end-start, 2), "s")
    print("- No. of docs indexed:", stats.numberOfDocuments)
    print("- No. of unique terms:", stats.numberOfUniqueTerms)
    print("- Total no. of terms:", stats.numberOfTokens)

    index_size = sum(f.stat().st_size for f in Path(index_path).glob('**/*') if f.is_file())
    index_size_mb = round(index_size / 1024**2, 1)

    print("- Index size: ", index_size_mb, "MB")
    print()

sys.exit(0)

# 2. Ranking models
# TODO: Split into 2:1 train/validation folds
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
#       - 
# TODO: Tune and run at least 1 learning-to-rank approach
#       - RankNet
#       - LambdaMART
#       - etc.
# 4. Evaluation
# TODO: use trec-eval: https://github.com/usnistgov/trec_eval
# TODO: Report MAP and NDCG at cut-offs of 5, 10, 20.
# TODO: possibly report more metrics
# TODO: report mean response times of your systems
# 4.1 Real-World Use Case
# TODO: Output submissions in the TREC run format