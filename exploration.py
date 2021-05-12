import sys
import json
import zipfile
import xmltodict
import numpy as np
import pandas as pd
from pprint import pprint

metadata = pd.read_csv("data/metadata.csv")
print("> Loading metadata")
print(metadata)

print("> Loading qrels_train")
qrels_train = pd.read_csv("data/qrels_train.txt", delimiter=' ', header=None)
qrels_train.rename(columns={
    0: "topic_number",
    1: "iteration",
    2: "cord_uid",
    3: "judgement",
}, inplace=True)
print(qrels_train)

print("> Loading topics_train.xml")
with open("data/topics_train.xml", "r") as f:
    fstr = "".join(f.readlines())
    topics_train = xmltodict.parse(fstr)

print("First topic:")
pprint(topics_train['topics']['topic'][0])
queries_train = list(map(lambda topic: topic['query'], topics_train['topics']['topic']))
print("Queries:")
pprint(queries_train)

print("> Loading embeddings...")
embeddings = pd.read_csv("data/cord_19_embeddings.zip", header=None)
embeddings.rename(columns={
    0: 'cord_uid'
}, inplace=True)
print(embeddings)

# Compute relevant statistics:
print("Nr. of articles:", len(metadata))
print("Article length:", len(metadata))

print("Nr. of assessed documents per topic:")
print(qrels_train.groupby('topic_number').count().iloc[:,0])

archive = zipfile.ZipFile("data/document_parses.zip", "r")
doc = archive.open("document_parses/pmc_json/PMC7091850.xml.json")
#doc = archive.open("document_parses/pdf_json/6b638fb47bfb48465ec020cfc22a6254d696dfc9.json")

data = json.load(doc)
pprint(data["body_text"])
# 1. Indexing
# TODO: Compute relevant statistics: 
#        - article length
#        - nr. of assessed documents per topic, average topic length
#        - etc.
# TODO: Index the dataset with Terrier, Indri, Elasticsearch, etc.
# TODO: Try diff. approaches to optimize index and see impact of each approach
#       - stopword removal
#       - stemming
#       - lemmatization
#       - etc.
# TODO: Extract information about index
#       - nr. of docs indexed
#       - nr. of unique terms
#       - total nr. of terms
#       - index size
#       - etc.
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