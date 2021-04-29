from series_prove_of_concept import EvaluationUtils

result_dir = "result_series_facets"

data_sets = [
    # # "classic_gutenberg",
    # "goodreads_genres",
    # # "goodreads_genres_short",
    # # "goodreads_genres_medium",
    # # "goodreads_genres_large",
    # "german_series",
    # "german_series_short",
    # "german_series_medium",
    # "german_series_large",
    # "german_books",
    # "dta",
    # "dta_short",
    # "dta_medium",
    # "dta_large",
    "litrec",
]
vectorization_algorithms = [
    "book2vec_concat",
    "book2vec_o_raw",
    "book2vec_wo_raw_concat",
    "book2vec_o_time",
    "book2vec_wo_time_concat",
    "book2vec_o_plot",
    "book2vec_wo_plot_concat",
    "book2vec_o_loc",
    "book2vec_wo_loc_concat",
    "book2vec_o_atm",
    "book2vec_wo_atm_concat",
    "book2vec_o_sty",
    "book2vec_wo_sty_concat",
]
filters = [
    # "no_filter",
    # "specific_words_moderate",
    "specific_words_strict"
]

task_names = [
    "AuthorTask",
    # "SeriesTask",
    # "GenreTask",
]

# EvaluationUtils.build_corpora(data_sets=data_sets,
#                               filters=filters)
#
#
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
                                       used_metrics=["ndcg",
                                                     "prec10", "f_prec10",
                                                     "rec10",  "f_rec10",
                                                     "f110", "f_f110"],
                                       filters=filters))

EvaluationUtils.latex_table(f"{result_dir}/z_table.csv", drop_columns=["Series_length", "Dataset", "Filter"])
