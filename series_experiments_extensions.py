from series_prove_of_concept import EvaluationUtils

result_dir = "result_series_extensions"

data_sets = [
    # "classic_gutenberg",
    "goodreads_genres",
    # "goodreads_genres_short",
    # "goodreads_genres_medium",
    # "goodreads_genres_large",
    # "german_series",
    # "german_series_short",
    # "german_series_medium",
    # "german_series_large",
    # "german_books",
    # "german_books_short",
    # "german_books_medium",
    # "german_books_large",
    # "dta",
    # "dta_short",
    # "dta_medium",
    # "dta_large",
    # "litrec",
    # "litrec_short",
    # "litrec_medium",
    # "litrec_large",
]
vectorization_algorithms = [
    # "bow",
    # "avg_wv2doc",
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
    # "doc2vec",
    # "book2vec",
    "book2vec_dbow",
    # "book2vec_dbow_pc",
    # "book2vec_dbow_concat",
    # "book2vec_chunk",
    # "book2vec_chunk_facet",
    # "book2vec_net",
    # "book2vec_net_only",
    # "book2vec_dbow",
    # "book2vec_adv",
    # "book2vec_adv_dbow",
    # "book2vec_window",
    #
    # "book2vec_window_0",
    # "book2vec_window_1",
    # "book2vec_window_2",
    # "book2vec_window_3",
    # "book2vec_window_4",
    # "book2vec_window_5",
    # "book2vec_window_10",
    # "book2vec_bert_concat",
    # "book2vec_bert_window_0_concat",
    # "book2vec_bert_window_1_concat",
    # "book2vec_bert_window_2_concat",
    # "book2vec_bert_window_3_concat",
    # "book2vec_bert_window_4_concat",
    # "book2vec_bert_window_5_concat",
    # "book2vec_bert_window_10_concat",

    # "book2vec_window_0_pca",
    # "book2vec_window_1_pca",
    # "book2vec_window_2_pca",
    # "book2vec_window_3_pca",
    # "book2vec_window_4_pca",
    # "book2vec_window_5_pca",
    # "book2vec_window_10_pca",

    # "book2vec_concat",
    # "book2vec_net_concat",
    # "book2vec_net_only_concat",
    # "book2vec_dbow_concat",
    # "book2vec_adv_concat",
    # "book2vec_adv_dbow_concat",
    # "book2vec_window_concat",


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
    "no_filter",
    # "specific_words_moderate",
    # "specific_words_strict"
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

# EvaluationUtils.run_evaluation(data_sets=data_sets,
#                                vectorization_algorithms=vectorization_algorithms,
#                                filters=filters,
#                                task_names=task_names,
#                                result_dir=result_dir)

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
