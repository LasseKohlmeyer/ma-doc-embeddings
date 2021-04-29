from experiments.series_prove_of_concept import EvaluationUtils

result_dir = "../result_bow"

data_sets = [
    # "goodreads_genres",
    # "german_series",
    "german_books",
]
vectorization_algorithms = [
    "bow",
    "doc2vec",
    "book2vec",
    "book2vec_concat"
]
filters = [
    "no_filter",
    "specific_words_moderate",
    "specific_words_strict"
]
task_names = [
    "AuthorTask",
    "SeriesTask",
    # "GenreTask",
]


EvaluationUtils.train_vecs(data_sets=data_sets,
                           vectorization_algorithms=vectorization_algorithms,
                           filters=filters)

EvaluationUtils.run_evaluation(data_sets=data_sets,
                               vectorization_algorithms=vectorization_algorithms,
                               filters=filters,
                               task_names=task_names,
                               result_dir=result_dir
                               )

print(
    EvaluationUtils.create_paper_table(f"{result_dir}/simple_series_experiment_table.csv", f"{result_dir}/z_table.csv",
                                       used_metrics=["ndcg", "f_prec", "f_prec01", "f_prec03", "f_prec05",
                                                     "f_prec10",
                                                     "length_metric"],
                                       filters=filters))
