from experiments.series_prove_of_concept import EvaluationUtils

result_dir = "../result_doc_window"

data_sets = [
    # "classic_gutenberg"
    # "goodreads_genres",
    # "german_series",
    "german_books",
]
vectorization_algorithms = [
    # "doc2vec_win1",
    # "doc2vec_win2",
    # "doc2vec_win3",
    "doc2vec_win4",
    # "doc2vec_win5",
    # "doc2vec_win7",
    # "doc2vec_win10",
    # "doc2vec_win15",
    # # "book2vec_win1",
    # # "book2vec_win1_concat",
    # # "book2vec_win2",
    # # "book2vec_win2_concat",
    # "book2vec_win3",
    # "book2vec_win3_concat",
    # "book2vec_win5",
    # "book2vec_win5_concat",
    # "book2vec",
    # "book2vec_concat",
    # "book2vec_win4",
    # "book2vec_win4_concat",
    # "book2vec_win5",
    # "book2vec_win5_concat",
    # "book2vec_win6",
    # "book2vec_win6_concat",
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

# print(
#     EvaluationUtils.create_paper_table(f"{result_dir}/simple_series_experiment_table.csv", f"{result_dir}/z_table_gb.csv",
#                                        used_metrics=["ndcg", "f_prec", "f_prec01", "f_prec03", "f_prec05",
#                                                      "f_prec10",
#                                                      "length_metric"],
#                                        filters=filters))

print(
    EvaluationUtils.create_paper_table(f"{result_dir}/simple_series_experiment_table.csv", f"{result_dir}/z_table.csv",
                                       used_metrics=["ndcg", "prec10", "f_prec10", "rec10", "f110"],
                                       filters=filters))
