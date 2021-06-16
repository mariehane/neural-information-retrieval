import time
import pandas as pd
import pyterrier as pt
from abc import ABC, abstractproperty, abstractmethod
from pathlib import Path

class Index(ABC): 
    @abstractmethod
    def __init__(self, name, path, overwrite=True, remove_stopwords=True, store_positions=False):
        raise NotImplementedError("Abstract method not overwritten!")

    @abstractmethod
    def load(self):
        raise NotImplementedError("Abstract method not overwritten!")

    @abstractmethod
    def create(self, text, *metadata):
        raise NotImplementedError("Abstract method not overwritten!")

    @abstractmethod
    def get_stats(self):
        raise NotImplementedError("Abstract method not overwritten!")

    @property
    @abstractmethod
    def exists(self):
        raise NotImplementedError("Abstract property not overwritten!")

class TerrierIndex(Index):

    def __init__(self, name, path, text, docno, metadata, overwrite=True, remove_stopwords=True, stem="porter", store_positions=False):
        self.name = name
        self.path = Path(path)
        self.text = text
        docno = docno.rename("docno")
        self.metadata = pd.concat([docno, *metadata], axis=1)

        indexer_path = str(self.path)
        if (not self.path.is_absolute()):
            indexer_path = "./" + str(self.path)

        if not pt.started():
            pt.init()

        self.indexer = pt.DFIndexer(indexer_path, overwrite=overwrite, blocks=store_positions)

        termpipelines = []
        if remove_stopwords:
            termpipelines.append("Stopwords")

        if stem=="porter":
            termpipelines.append("PorterStemmer")
        elif stem=="snowball":
            termpipelines.append("SnowballStemmer")
        termpipelines = ",".join(termpipelines)
        self.indexer.setProperty("termpipelines", termpipelines)
        
        #if tokenizer != "EnglishTokeniser":
        #    self.indexer.setProperty("tokeniser", index_dict["tokeniser"])

        # TODO: Add support for other preprocessing


    def load(self):
        index_ref = self.path / "data.properties"
        if (not index_ref.is_absolute()):
            index_ref = "./" + str(index_ref)
        else:
            index_ref = str(index_ref)

        self.index = pt.IndexFactory.of(index_ref)
    
    def create(self):
        start = time.time()
        index_ref = self.indexer.index(self.text, self.metadata)
        self.index = pt.IndexFactory.of(index_ref)
        end = time.time()
        return end-start

    def get_stats(self):
        if self.index is None:
            raise RuntimeError("Index has not been loaded/created!")

        stats = self.index.getCollectionStatistics()
        n_docs = stats.numberOfDocuments
        n_unique_terms = stats.numberOfUniqueTerms
        n_tokens = stats.numberOfTokens
        index_size = sum(f.stat().st_size for f in self.path.glob('**/*') if f.is_file())
        index_size_mb = round(index_size / 1024**2, 1)

        return n_docs, n_unique_terms, n_tokens, index_size_mb

    def exists(self):
        return (self.path / "data.properties").exists()
    
    def __str__(self):
        return f"TerrierIndex({self.index})"