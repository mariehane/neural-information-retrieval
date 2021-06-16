import pytrec_eval
from pathlib import Path

def output_run(model, model_name, topics_df, student_id, path, n_docs_per_topic=1000, filter_out=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tf_run = []
    for _, row in topics_df.iterrows():
        qid, query = row
        res_df = model.search(query)
        if filter_out is not None:
            mask = ~res_df.docno.isin(filter_out)
            res_df = res_df[mask]
        
        for i, res_row in res_df[:1000].iterrows():
            docno = res_row["docno"]
            if "rank" in res_row:
                rank = res_row["rank"]
            else:
                rank = i
            score = res_row["score"]
            query = res_row["query"]
            row_str = f"{qid} 0 {docno} {rank} {score} {student_id}-{model_name}"
            tf_run.append(row_str)
    with open(path, "w") as f:
        for l in tf_run:
            f.write(l + "\n")

def qrels_to_dict(qrels_df):
    qrels_dict = dict()
    for _, r in qrels_df.iterrows():
        qid, docno, label, _ = r
        if qid not in qrels_dict:
            qrels_dict[qid] = dict()
        qrels_dict[qid][docno] = int(label)
    return qrels_dict

def evaluate(run_file, qrels, metrics={"map", "ndcg", "ndcg_cut_10", "P_10"}):
    pass
    #evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

    #with open(run_path, 'r') as f_run:
    #    tf_run = pytrec_eval.parse_run(f_run)

    ## compute average across topics
    #for m in metrics:
    #    print(m, '\t', pytrec_eval.compute_aggregated_measure(m, tf_metric2vals[m]))