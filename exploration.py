import sys

import json
import zipfile
import xmltodict
import numpy as np
import pandas as pd
from pprint import pprint
from pathlib import Path

from data import Cord19Dataset

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

print("Avg. no. of assessed documents per topic:", qrels_train.groupby('topic_number').count().iloc[:,0].mean())


archive = zipfile.ZipFile("data/document_parses.zip", "r")
doc = archive.open("document_parses/pmc_json/PMC7091850.xml.json")
#doc = archive.open("document_parses/pdf_json/6b638fb47bfb48465ec020cfc22a6254d696dfc9.json")
doc_json = json.load(doc)
text_elems = [ elem["text"] for elem in doc_json["body_text"] ]
text = "\n".join(text_elems)
print("Length of chosen article:", len(text))

# Check if there are any duplicates where only one has text
dataset = Cord19Dataset(base_dir='data', download=True)
path_df = Path("data/df.csv")
if path_df.exists():
    print("> Loading simple dataframe...")
    df = pd.read_csv(str(path_df))
else:
    print("> Creating simple dataframe...")
    df = dataset.get_dataframe(len(dataset.metadata), show_progressbar=True)

# cleanup df
df.replace('None', np.nan, inplace=True)
df["text"] = df["text"].apply(lambda s: s.strip() if s is not np.nan else np.nan)
df["text"] = df["text"].apply(lambda s: np.nan if s == "" else s)

df_duplicates = df[df.duplicated(subset='cord_uid', keep=False)].sort_values(by="cord_uid")
for uid in df_duplicates.cord_uid.unique():
    rows = df[df.cord_uid == uid] 
    contains_nans = rows[["abstract", "text"]].isna().any().any()
    if contains_nans:
        missing = rows[["abstract", "text"]].isna().sum(axis=0)
        total = len(rows)
        print(uid, f"{missing['abstract']}/{total} missing abstract, {missing['text']}/{total} text")

# basic duplicate removal
#df = df.drop_duplicates(subset="cord_uid")

# duplicate removal that prefers rows with filled title, abstract and text
df_sorted = df.sort_values(by="title").sort_values(by="abstract").sort_values(by="text")
df = df_sorted.drop_duplicates(subset="cord_uid", keep="first")

# Check if any docs in the qrels have completely empty fields
df_merged = dataset.qrels_train.merge(df, on='cord_uid')
# title + abstract + text: 0 docs
df_merged[df_merged[["title", "abstract", "text"]].isna().all(axis=1)]
# abstract + text: 1302 docs
df_merged[df_merged[["abstract", "text"]].isna().all(axis=1)]
# abstract: 6760 docs
df_merged[df_merged[["abstract"]].isna().all(axis=1)]
# text: 8166 docs
df_merged[df_merged[["text"]].isna().all(axis=1)]
# title: 0 docs
df_merged[df_merged[["title"]].isna().all(axis=1)]
