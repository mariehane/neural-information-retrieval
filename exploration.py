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

print("Nr. of assessed documents per topic:")
print(qrels_train.groupby('topic_number').count().iloc[:,0])

archive = zipfile.ZipFile("data/document_parses.zip", "r")
doc = archive.open("document_parses/pmc_json/PMC7091850.xml.json")
#doc = archive.open("document_parses/pdf_json/6b638fb47bfb48465ec020cfc22a6254d696dfc9.json")
doc_json = json.load(doc)
text_elems = [ elem["text"] for elem in doc_json["body_text"] ]
text = "\n".join(text_elems)
print("Length of chosen article:", len(text))
