import pytrec_eval

def output_run(model, model_name, topics_df, path):
    tf_run = []
    for _, row in topics_df.iterrows():
        qid, query = row
        res_df = model.search(query)
        for _, res_row in res_df.iterrows():
            _, _, docno, rank, score, query = res_row
            row_str = f"{qid} 0 {docno} {rank} {score} {model_name}"
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

def evaluate(run_file, qrels, metrics={"map", "ndcg", "ndcg_cut_10", "P_10"})
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

    with open(run_path, 'r') as f_run:
        tf_run = pytrec_eval.parse_run(f_run)

# compute average across topics
for m in metrics:
    print(m, '\t', pytrec_eval.compute_aggregated_measure(m, tf_metric2vals[m]))