from series_prove_of_concept import EvaluationUtils

result_dir = "result_doc_window"

data_sets = [
    # "classic_gutenberg"
    # "goodreads_genres",
    # "german_series",
    "german_books",
]
vectorization_algorithms = [
    "doc2vec_dim50",
    "doc2vec_dim100",
    "doc2vec_dim300",
    "doc2vec_dim500",
    "doc2vec_dim700",
    "doc2vec_dim900",
    "book2vec_dim50",
    "book2vec_dim50_concat",
    "book2vec_dim100",
    "book2vec_dim100_concat",
    "book2vec_dim300",
    "book2vec_dim300_concat",
    "book2vec_dim500",
    "book2vec_dim500_concat",
    "book2vec_dim700",
    "book2vec_dim700_concat",
    "book2vec_dim900",
    "book2vec_dim900_concat",
    # "book2vec",
    # "book2vec_concat",
]
filters = [
    "no_filter",
    # "specific_words_moderate",
    # "specific_words_strict"
]

task_names = [
    "AuthorTask",
    "SeriesTask",
    # "GenreTask",
]

EvaluationUtils.build_corpora(data_sets=data_sets,
                              filters=filters)


EvaluationUtils.train_vecs(data_sets=data_sets,
                           vectorization_algorithms=vectorization_algorithms,
                           filters=filters)

EvaluationUtils.run_evaluation(data_sets=data_sets,
                               vectorization_algorithms=vectorization_algorithms,
                               filters=filters,
                               task_names=task_names,
                               result_dir=result_dir)

print(
    EvaluationUtils.create_paper_table(f"{result_dir}/simple_series_experiment_table.csv", f"{result_dir}/z_table.csv",
                                       used_metrics=["ndcg", "f_prec", "f_prec01", "f_prec03", "f_prec05",
                                                     "f_prec10",
                                                     "length_metric"],
                                       filters=filters))
