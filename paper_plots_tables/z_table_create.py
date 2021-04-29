from experiments.series_prove_of_concept import EvaluationUtils

print(EvaluationUtils.create_paper_table("results/series_experiment_table2202_german_books.csv",
                                         "../results/z_table.csv",
                                         used_metrics=["ndcg", "f_prec", "f_prec01", "f_prec03", "f_prec05",
                                                       "f_prec10",
                                                       "length_metric"],
                                         filters=["no_filter",
                                                  "specific_words_moderate",
                                                  "specific_words_strict"
                                                  ]))