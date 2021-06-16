import optuna
import torch
import gensim
import gensim.downloader as gensim_api
import pandas as pd
import pyterrier as pt
import lightgbm as lgbm
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import ExtraTreesRegressor

if not pt.started():
    pt.init() # required to access pt.batchretrieve

def tune_bm25(topics, qrels, index, n_trials=100, metric='map', verbose=False, show_progressbar=False, seed=None):
    def objective(trial):
        c = trial.suggest_uniform('c', 0, 10)
        k_1 = trial.suggest_uniform('k_1', 0, 10)
        k_3 = trial.suggest_uniform('k_3', 0, 10)

        bm25 = pt.BatchRetrieve(index.index, wmodel="BM25", controls={"c": c, "bm25.k_1": k_1, "bm25.k_3": k_3})
        results = pt.Experiment(
            retr_systems=[bm25],
            names=['BM25'],
            topics=topics,
            qrels=qrels,
            eval_metrics=[metric])

        return results.loc[0, metric]

    if verbose:
        print()
        print(f"> Running optuna on index <{index.name}> with metric <{metric}>")

    sampler = None
    if seed is not None:
        sampler = optuna.samplers.TPESampler(seed=seed)
    
    study = optuna.create_study(study_name="BM25 Tuning", direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=100, show_progress_bar=show_progressbar)

    if verbose:
        print("- Best params:")
        print(study.best_params)

    return pt.BatchRetrieve(index.index, wmodel="BM25", 
                            controls={"c": study.best_params["c"], 
                                    "bm25.k_1": study.best_params["k_1"],
                                    "bm25.k_3": study.best_params["k_3"]})

class GensimQueryExpander(pt.batchretrieve.BatchRetrieveBase):
    """Gensim query expander that slots into PyTerrier's evaluation framework
    
    Query expansion code is heavily based on code from the Lab 4 notebook by Emanuele Bugliarello
    """
    problematic_words = set([".", ",", "n't", "'ll", "hiv/aids"]) # these words will cause a crash if parsed by PyTerrier

    def __init__(self, next, k=10, model='conceptnet-numberbatch-17-06-300'):
        """
            next : PyTerrier retrieval system to apply after query expansion
            k    : Limit for how many words to expand by
        """
        self.next = next
        self.model = gensim_api.load(model)
        self.k = k
    
    def transform(self, topics):
        topics_qe = topics.copy()
        for i in range(len(topics_qe)):
            query = topics_qe.iloc[i]['query']
            expanded = []
            for word in query.split(' '):
                if word in self.model:
                    expanded_words = [pair[0] for pair in self.model.most_similar(word, topn=self.k)]
                    expanded_words.append(word)
                    expanded.append(expanded_words)

            expanded_str = gensim.parsing.preprocessing.remove_stopwords(" ".join([e for l in expanded for e in l if e not in self.problematic_words]))
            topics_qe.iloc[i]['query'] = expanded_str

        # feed expanded queries to next system in pipeline
        return self.next.transform(topics_qe)


class DistilBertLM(pt.batchretrieve.BatchRetrieveBase):
    """DistilBert Re-ranking model that slots into PyTerrier's evaluation framework

    The actual re-reranking code is heavily based on code from the Lab 6 notebook by Emanuele Bugliarello
    """
    def __init__(self, base, index):
        """
        """
        self.base = base
        self.metadata = pd.concat([index.text, index.metadata], axis=1)
        self.text_column = index.text.name
        self.model = SentenceTransformer("msmarco-distilbert-base-v3")
    
    def transform(self, topics):
        df = self.base.transform(topics)

        df_augmented = df.merge(self.metadata, on="docno") # augment with text data

        query_embedding = self.model.encode(df_augmented["query"])
        docs_embedding = self.model.encode(df_augmented[self.text_column])
        cos_sims = util.pytorch_cos_sim(query_embedding, docs_embedding)
        cos_sims = torch.diag(cos_sims)

        df_result = df_augmented[df.columns].copy() # remove augmented data
        df_result["score"] = cos_sims
        df_result.drop(columns="rank", inplace=True)
        return df_result


def create_models(abstract_index, text_index, topics, qrels, seed=None, verbose=False):
    """
    """
    tf     = pt.BatchRetrieve(text_index.index, wmodel="Tf")
    tfidf  = pt.BatchRetrieve(text_index.index, wmodel="TF_IDF")
    bm25v1 = pt.BatchRetrieve(text_index.index, wmodel="BM25")
    bm25v2 = pt.BatchRetrieve(text_index.index, wmodel="BM25", controls={"c": 0.1, "bm25.k_1": 2.0, "bm25.k_3": 10})
    bm25v3 = pt.BatchRetrieve(text_index.index, wmodel="BM25", controls={"c": 8, "bm25.k_1": 1.4, "bm25.k_3": 10})

    bm25_best_abstracts = tune_bm25(topics, qrels, abstract_index, n_trials=100, verbose=verbose, show_progressbar=True, seed=seed)
    bm25_best_texts = tune_bm25(topics, qrels, text_index, n_trials=100, verbose=verbose, show_progressbar=True, seed=seed)

    bm25_best_abstracts_ndcg = tune_bm25(topics, qrels, text_index, metric='ndcg', n_trials=100, verbose=verbose, show_progressbar=True, seed=seed)
    bm25_best_texts_ndcg = tune_bm25(topics, qrels, text_index, metric='ndcg', n_trials=100, verbose=verbose, show_progressbar=True, seed=seed)

    # Dirichlet Language model 
    lm_dirichlet = pt.BatchRetrieve(text_index.index, wmodel="DirichletLM")
    index1_n_docs, _, index1_n_tokens, _ = text_index.get_stats()
    index1_avg_length = index1_n_tokens / index1_n_docs
    lm_dirichlet_avg = pt.BatchRetrieve(text_index.index, wmodel="DirichletLM", controls={"c": index1_avg_length})

    # Query rewriting using Sequential Dependence Model (SDM)
    #sdm = pt.rewrite.SequentialDependence()
    #sdm_bm25v1 = sdm >> bm25v1
    #sdm_bm25_best_texts = sdm >> bm25_best_texts

    # Query rewriting using Pseudo-relevance feedback
    rm3 = pt.rewrite.RM3(text_index.index)
    rm3_pipe = bm25v1 >> rm3 >> bm25v1

    # Query expansion using GloVE embeddings
    if verbose:
        print("> Creating Query Expander with GloVE embeddings")
    glove_qe = GensimQueryExpander(next=bm25_best_texts, model='glove-wiki-gigaword-100')

    # note: GloVE doesn't have embeddings for e.g. 'COVID-19'
    # FastText should be better:
    if verbose:
        print("> Creating Query Expander with FastText embeddings")
    fasttext_qe = GensimQueryExpander(next=bm25_best_texts, model='fasttext-wiki-news-subwords-300')

    # Reranking with contextual BERT embeddings
    if verbose:
        print("> Creating DistilBERT models")
    lm_bert = DistilBertLM(base=bm25v1 % 100, index=text_index) #text=indexes[1].text, metadata=indexes[1].metadata)
    lm_bert_200 = DistilBertLM(base=bm25v1 % 200, index=text_index) #text=indexes[1].text, metadata=indexes[1].metadata)

    # Learning to rank with ExtraTrees
    if verbose:
        print("> Training ExtraTrees...")
    ltr_feats = (bm25v1 % 100) >> (tfidf ** lm_dirichlet ** bm25_best_texts ** bm25_best_abstracts)
    rf = ExtraTreesRegressor(n_estimators=100, verbose=1, random_state=seed, n_jobs=-1)
    rf_pipe = ltr_feats >> pt.ltr.apply_learned_model(rf)
    rf_pipe.fit(topics, qrels)

    #print("> Training LambdaMART")
    #lmart_1 = lgbm.LGBMRanker(
    #    task="train",
    #    silent=False,
    #    min_data_in_leaf=1,
    #    min_sum_hessian_in_leaf=1,
    #    max_bin=255,
    #    num_leaves=31,
    #    objective='lambdarank',
    #    metric='ndcg',
    #    ngcdg_eval_at=[10],
    #    ndcg_at=[10],
    #    eval_at=[10],
    #    learning_rate=0.1,
    #    importance_type='gain',
    #    num_iterations=100,
    #    early_stopping_rounds=5
    #)

    #lmart_x_pipe = ltr_feats >> pt.ltr.apply_learned_model(lmart_1, form="ltr", fit_kwargs={'eval_at': [10]})
    #lmart_x_pipe.fit(topics, qrels, topics, qrels_valid)

    # CombSUM rank fusion
    bm25_abstract_and_text = 1.0*bm25_best_abstracts + 1.0*bm25_best_texts

    # More advanced combinations
    advanced_bert = DistilBertLM(base=bm25_abstract_and_text % 200, index=text_index)

    dirichlet_bm25 = 1.0*bm25_abstract_and_text + 1.0*lm_dirichlet
    # Conceptnet Expansion >> (bm25_abstract + bm25_text + dirichlet) >> DistilBERT reranking of Top-100
    #advanced_pipe = GensimQueryExpander(next=DistilBertLM(base=dirichlet_bm25 % 100, index=indexes[1]))

    models = {
        'TF': tf,
        'TF-IDF': tfidf,
        'BM25v1': bm25v1,
        'BM25v2': bm25v2,
        'BM25v3': bm25v3,
        'Tuned BM25 (abstracts)': bm25_best_abstracts,
        'Tuned BM25 (texts)': bm25_best_texts,
        'Tuned BM25 for NDCG (abstracts)': bm25_best_abstracts_ndcg,
        'Tuned BM25 for NDCG (texts)': bm25_best_texts_ndcg,
        'Dirichlet LM': lm_dirichlet,
        'Dirichlet LM (mu = avg doc len)': lm_dirichlet_avg,
        #'SDM >> BM25v1': sdm_bm25v1,
        #'SDM >> Tuned BM25 texts': sdm_bm25_best_texts,
        'RM3': rm3_pipe,
        'GloVE QE >> Tuned BM25 abstracts': glove_qe,
        'FastText QE >> Tuned BM25 abstracts': fasttext_qe,
        'BM25v1 Top-100': bm25v1 % 100,
        'BM25v1 Top-100 > DistilBERT': lm_bert,
        'BM25v1 Top-200 > DistilBERT': lm_bert_200,
        'BM25 Top-100 > ExtraTrees': rf_pipe,
        #'LambdaMART': lmart_x_pipe,
        'CombSUM of tuned BM25s abstract+text': bm25_abstract_and_text,
        'CombSUM of tuned BM25s and Dirichlet LM': dirichlet_bm25,
        'abstract+text Top 200 >> DistilBERT': advanced_bert,
        #'Advanced Pipe': advanced_pipe
    }
    return models
