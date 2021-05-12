import sys
import json
import zipfile
import xmltodict
import numpy as np
import pandas as pd
from pprint import pprint

from data import Cord19Dataset

print("> Loading dataset")
dataset = Cord19Dataset(base_dir='data', download=True)
dataset.get_paper_text(paper_index=0)
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