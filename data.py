import zipfile
import json
import xmltodict
import tqdm
import numpy as np
import pandas as pd

from pathlib import Path

class Cord19Dataset():
    def __init__(self, base_dir, download=False):
        self.base_dir = Path(base_dir)
        if (not self.base_dir.exists()):
            if (not download):
                raise RuntimeError("CORD-19 Dataset not found!")

            print(f"Directory for CORD-19 Dataset ('{base_dir}') does not exist. Downloading...")
            Cord19Dataset.download(base_dir)

        self.cord19_embeddings_path = self.base_dir / 'cord_19_embeddings.zip'
        self.document_parses_path = self.base_dir / 'document_parses.zip'
        self.metadata_path = self.base_dir / 'metadata.csv'
        self.qrels_train_path = self.base_dir / 'qrels_train.txt'
        self.topics_train_path = self.base_dir / 'topics_train.xml'
        self.topics_test_path = self.base_dir / 'topics_test.xml'

        self.metadata = pd.read_csv(self.metadata_path)
        self.qrels_train = pd.read_csv(self.qrels_train_path, delimiter=' ', header=None)
        self.qrels_train.rename(columns={
            0: "topic_number",
            1: "iteration",
            2: "cord_uid",
            3: "judgement",
        }, inplace=True)

        self.topics_train = self._get_topics_df(self.topics_train_path)
        self.topics_test = self._get_topics_df(self.topics_test_path)

        self.document_parses_zip = zipfile.ZipFile(self.document_parses_path, "r")

        #embeddings = pd.read_csv(self.cord19_embeddings_path, header=None)
        #embeddings.rename(columns={
        #    0: 'cord_uid'
        #}, inplace=True)

    def _get_topics_df(self, topics_xml_path):
        with open(topics_xml_path, "r") as f:
            xml_dict = xmltodict.parse(f.read())

            topics_df = []
            for topic in xml_dict["topics"]["topic"]:
                topics_df.append([topic["@number"], topic["query"], topic["question"], topic["narrative"]])

            topics_df = pd.DataFrame(topics_df, columns=["topic_number", "query", "question", "narrative"])
            #topics_df.set_index("topic_number", inplace=True)

            return topics_df

    def get_paper_text(self, paper_index):
        path = self.metadata.loc[paper_index, "pmc_json_files"]
        if (type(path) is not str):
            path = self.metadata.loc[paper_index, "pdf_json_files"]
            if (type(path) is not str):
                raise RuntimeError(f"Paper {paper_index} does not have any PMC/PDF json parses associated with it!")
        if (";" in path):
            path = path.split(";")[0]

        with self.document_parses_zip.open(path) as doc:
            data = json.load(doc)
            text_elems = [ elem["text"] for elem in data["body_text"] ]
            text = "\n".join(text_elems)
            return text

    def get_dataframe(self, n_papers, show_progressbar=False):
        df = self.metadata.loc[:n_papers, ["cord_uid","title","abstract","publish_time","journal"]]

        def try_get_text(i):
            try:
                return self.get_paper_text(i)
            except RuntimeError:
                return None

        iterator = map(try_get_text, range(n_papers))
        if show_progressbar:
            iterator = tqdm.tqdm(iterator, total=n_papers)
        df["text"] = pd.DataFrame(iterator)

        df = df.replace({np.nan: None})
        df = df.astype(str)
        return df

    @staticmethod
    def download(path):
        pass # TODO: download from Git LFS or directly from CORD19 (may be missing qrels)