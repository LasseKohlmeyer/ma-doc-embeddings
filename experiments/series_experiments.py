from series_prove_of_concept import EvaluationUtils

result_dir = "../result_series"

data_sets = [
    # "classic_gutenberg"
    "goodreads_genres",
    # "goodreads_genres_short",
    # "goodreads_genres_medium",
    # "goodreads_genres_large",
    # "german_series",
    # "german_books",
    # "dta",
    # "litrec",
]
vectorization_algorithms = [
    "bow",
    "avg_wv2doc",
    # "avg_wv2doc_restrict10000",
    # "doc2vec",
    # "doc2vec_chunk",
    # "psif",
    # "bert_pt",
    # # "bert_pt_chunk",
    # # "bert_sentence_based_1000_pt",
    # "roberta_pt",
    # # "roberta_pt_chunk",
    # # "roberta_sentence_based_1000_pt",
    # "xlm_pt",
    # # "xlm_pt_chunk",
    # # "xlm_sentence_based_1000_pt",
    # "book2vec",
    # "book2vec_avg",
    # "book2vec_concat",
    # "book2vec_pca",
    # "book2vec_auto",
    #
    # # "book2vec_net_only_concat",
    # # "book2vec_auto",
    # # "book2vec_o_raw",
    # # "book2vec_o_time",
    # # "book2vec_o_plot",
    # # "book2vec_o_loc",
    # # "book2vec_o_atm",
    # # "book2vec_o_sty",
    # # "book2vec_wo_raw_concat",
    # # "book2vec_net_concat",
    #
    # # "book2vec_dbow_concat",
    # # "bookwvec_window"
    # # "book2vec_adv",
    # # "book2vec_adv_concat",
]
filters = [
    # "no_filter",
    # "specific_words_moderate",
    "specific_words_strict"
]

task_names = [
    "AuthorTask",
    # "SeriesTask",
    "GenreTask",
]

# EvaluationUtils.build_corpora(data_sets=data_sets,
#                               filters=filters)
#
#
# EvaluationUtils.train_vecs(data_sets=data_sets,
#                            vectorization_algorithms=vectorization_algorithms,
#                            filters=filters)

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
