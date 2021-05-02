from collections import defaultdict

from scipy.stats import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from lib2vec.corpus_structure import DataHandler
from lib2vec.vectorization_utils import Vectorization
import sklearn as sk
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.contingency_tables import mcnemar


def cross_val_scores_weighted(model, X, y, weights, cv=5, metrics=[sk.metrics.accuracy_score]):
    kf = sk.model_selection.KFold(n_splits=cv)
    kf.get_n_splits(X)
    scores = [[] for metric in metrics]
    for train_index, test_index in kf.split(X):
        model_clone = sk.base.clone(model)
        print(train_index, test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]
        model_clone.fit(X_train,y_train,sample_weight=weights_train)
        y_pred = model_clone.predict(X_test)
        for i, metric in enumerate(metrics):
            score = metric(y_test, y_pred, sample_weight = weights_test)
            scores[i].append(score)
    return scores


def chi_square_test(input1, input2):
    truth1 = [0.5 for e in input1]
    truth2 = [0.5 for e in input2]
    print('---')
    print(stats.chisquare(input1, truth1))
    print(stats.chisquare(input2, truth2))
    print('---')


def mcnemar_sig_text(table):
    # define contingency table
    # calculate mcnemar test
    exact = False
    correction = True
    for line in table:
        for e in line:
            if e < 25:
                exact = True
                correction = False
    result = mcnemar(table, exact=exact, correction=correction)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print(f'Classifiers have same proportions of errors (fail to reject H0), {result.pvalue}')
    else:
        print(f'Classifiers have different proportions of errors (reject H0), {result.pvalue}')


def success_prediction_task(data_set_name: str, success_dict, vector_names):
    result_tuples = []
    correctness_table = defaultdict(list)
    for vectorization_algorithm in vector_names:
        majority_class = None
        if vectorization_algorithm == "majority_class":
            majority_class = vectorization_algorithm
            vectorization_algorithm = "doc2vec"

        vec_path = Vectorization.build_vec_file_name("all",
                                                     "no_limit",
                                                     data_set_name,
                                                     "no_filter",
                                                     vectorization_algorithm,
                                                     "real",
                                                     allow_combination=True)
        vectors, summation_method = Vectorization.my_load_doc2vec_format(vec_path)

        x = []
        y = []
        doc_ids = []
        k_fold_cross_val = None
        for doctag in vectors.docvecs.doctags:
            try:
                if summation_method and f"_{summation_method}" in str(doctag):
                    x.append(vectors.docvecs[doctag])
                    # success = success_dict[doctag.replace(f"_{summation_method}", "")]
                    # if success == "failure":
                    #     success = 0
                    # else:
                    #     success = 1
                    # y.append(success)
                    y.append(0 if success_dict[doctag.replace(f"_{summation_method}", "")] == "failure" else 1)
                    doc_ids.append(doctag.replace(f"_{summation_method}", ""))

                elif not summation_method and str(doctag)[-1].isdigit():
                    doc_splitted = doctag.split("_")
                    if len(doc_splitted) > 1 and doc_splitted[-1][-1].isdigit() and doc_splitted[-2][-1].isdigit():
                        pass
                    else:
                        x.append(vectors.docvecs[doctag])
                        y.append(0 if success_dict[doctag] == "failure" else 1)
                        doc_ids.append(doctag)
                else:
                    pass
            except KeyError:
                pass

        counter = defaultdict(lambda: 0)
        print(len(y))
        for truth_val in y:
            counter[truth_val] += 1
        # print(len(doc_ids), len(y))
        # print(doc_ids)

        # print(y)
        classifiers = [
            # RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
            sk.svm.LinearSVC(max_iter=10000, class_weight="balanced", dual=False, random_state=42, C=1.5),
            sk.svm.SVC(max_iter=10000, class_weight="balanced", kernel="rbf", random_state=42, C=0.5),
            sk.svm.NuSVC(max_iter=10000, class_weight="balanced", kernel="rbf", gamma="scale", random_state=42, nu=0.5),
            # MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(10, 4), random_state=42, max_iter=10000),
            # LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=1000),
        ]

        # classifiers = [make_pipeline(StandardScaler(), classifier) for classifier in classifiers]
        for classifier in classifiers:
            pipeline_classifier = make_pipeline(StandardScaler(), classifier)
            if k_fold_cross_val:
                weights = [counter[pred] for pred in y]
                # results = cross_val_scores_weighted(classifier, x, y, weights, cv=k_fold_cross_val,
                #                                     metrics=[f1_score, precision_score,
                #                                              recall_score, f1_score])

                scoring = {
                    'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score),
                    'recall': make_scorer(recall_score),
                    'f1_score': make_scorer(f1_score)
                }
                kfold = sk.model_selection.KFold(n_splits=k_fold_cross_val, random_state=42, shuffle=True)
                results = sk.model_selection.cross_validate(estimator=pipeline_classifier,
                                                            X=x,
                                                            y=y,
                                                            cv=kfold,
                                                            scoring=scoring)

                result_tuples.append((classifier.__class__.__name__,
                                      vectorization_algorithm,
                                      np.mean(results["test_f1_score"]),
                                      np.mean(results["test_precision"]),
                                      np.mean(results["test_recall"]),
                                      np.mean(results["test_accuracy"])))
            else:
                nr_iterations = 100
                f1s = []
                for i in range(nr_iterations):
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=i)
                    pipeline_classifier.fit(x_train, y_train)
                    y_pred = pipeline_classifier.predict(x_test)

                    if majority_class:
                        y_pred = [1 for y in y_pred]
                        vectorization_algorithm = majority_class

                    # weights = [counter[pred] for pred in y_test]
                    correctness_table[f"{vectorization_algorithm}-{classifier.__class__.__name__}"]\
                        .extend([pred == truth for pred, truth in zip(y_pred, y_test)])
                    f1s.append(f1_score(y_test, y_pred, average="weighted", pos_label=None))
                    # print(weights)
                    # print(vectorization_algorithm, round(classifier.score(x_test, y_test), 4))

                if len(f1s) == 0:
                    f1s.append(0)
                result_tuples.append((classifier.__class__.__name__,
                                      vectorization_algorithm,
                                      np.mean(f1s),
                                      # precision_score(y_test, y_pred, average="weighted"),
                                      # recall_score(y_test, y_pred,  average="weighted"),
                                      # accuracy_score(y_test, y_pred)
                                      )
                                     )

    print(correctness_table.keys())
    classifiers = [
        "LinearSVC",
        "SVC",
        "NuSVC"
    ]
    for cl in classifiers:
        algo1 = "xlm_pt"
        algo2 = "book2vec_concat"
        true_true = 0
        true_false = 0
        false_true = 0
        false_false = 0
        for e1, e2 in zip(correctness_table[f"{algo1}-{cl}"], correctness_table[f"{algo2}-{cl}"]):
            if e1 and e2:
                true_true += 1
            elif e1 and not e2:
                true_false += 1
            elif not e1 and e2:
                false_true += 1
            elif not e1 and not e2:
                false_false += 1
            else:
                pass
        table = [[true_true, true_false],
                 [false_true, false_false]]
        print(table)
        print(cl)
        mcnemar_sig_text(table)

    df = pd.DataFrame(result_tuples, columns=["Classifier", "Algorithm", "Weighted F1",
                                              # "Precision", "Recall",
                                              # "Accuracy"
                                              ])
    print(df)
    df = df.pivot(index='Algorithm', columns='Classifier', values='Weighted F1')
    df.to_csv("../results/book_success_prediction/eval_scores.csv", index=True)
    print(df)
    print(df.to_latex())

    # df = df.round(4)


def genre_prediction_task(data_set_name: str, genre_dict, vector_names):
    result_tuples = []
    for vectorization_algorithm in vector_names:

        vec_path = Vectorization.build_vec_file_name("all",
                                                     "no_limit",
                                                     data_set_name,
                                                     "no_filter",
                                                     vectorization_algorithm,
                                                     "real",
                                                     allow_combination=True)
        vectors, summation_method = Vectorization.my_load_doc2vec_format(vec_path)

        x = []
        y = []

        k_fold_cross_val = None

        for doctag in vectors.docvecs.doctags:
            if summation_method and f"_{summation_method}" in str(doctag):
                x.append(vectors.docvecs[doctag])
                # success = success_dict[doctag.replace(f"_{summation_method}", "")]
                # if success == "failure":
                #     success = 0
                # else:
                #     success = 1
                # y.append(success)
                y.append(genre_dict[doctag.replace(f"_{summation_method}", "")])

            else:
                if str(doctag)[-1].isdigit():
                    x.append(vectors.docvecs[doctag])
                    y.append(genre_dict[doctag])

        # print(y)
        classifiers = [
            # RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
            # sk.svm.LinearSVC(max_iter=10000, class_weight="balanced", dual=False, random_state=42),
            sk.svm.LinearSVC(max_iter=20000, class_weight="balanced", dual=True, random_state=42),
            # sk.svm.LinearSVC(max_iter=10000, class_weight="balanced", dual=True, random_state=42, C=5),
            # sk.svm.LinearSVC(max_iter=10000, dual=False, random_state=42),
            # sk.svm.LinearSVC(max_iter=10000, dual=True, random_state=42),
            # MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42, max_iter=10000),
            # LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=1000)
        ]
        for classifier in classifiers:
            if k_fold_cross_val:
                scoring = {
                    'accuracy': make_scorer(accuracy_score),
                    'precision': make_scorer(precision_score),
                    'recall': make_scorer(recall_score),
                    'f1_score': make_scorer(f1_score)
                }

                kfold = sk.model_selection.KFold(n_splits=k_fold_cross_val, random_state=42, shuffle=True)

                results = sk.model_selection.cross_validate(estimator=classifier,
                                                            X=x,
                                                            y=y,
                                                            cv=kfold,
                                                            scoring=scoring)

                result_tuples.append((classifier.__class__.__name__,
                                      vectorization_algorithm,
                                      np.mean(results["test_f1_score"]),
                                      np.mean(results["test_precision"]),
                                      np.mean(results["test_recall"]),
                                      np.mean(results["test_accuracy"])))
            else:
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
                classifier.fit(x_train, y_train)
                y_pred = classifier.predict(x_test)
                # print(z)
                # print(vectorization_algorithm, round(classifier.score(x_test, y_test), 4))
                result_tuples.append((classifier.__class__.__name__,
                                      vectorization_algorithm,
                                      f1_score(y_test, y_pred, average='weighted'),
                                      precision_score(y_test, y_pred, average='weighted'),
                                      recall_score(y_test, y_pred, average='weighted'),
                                      accuracy_score(y_test, y_pred)))
    df = pd.DataFrame(result_tuples, columns=["Classifier", "Algorithm", "F1", "Precision", "Recall", "Accuracy"])
    df = df.pivot(index='Algorithm', columns=['Classifier'], values='F1')
    df.to_csv("results/book_success_prediction/genre_eval_scores.csv", index=True)
    print(df)
    print(df.to_latex())


if __name__ == '__main__':
    c = DataHandler.load_maharjan_goodreads()
    algorithms = [
        # # "majority_class",
        # "avg_wv2doc",
        # "bow",
        # "avg_wv2doc_restrict10000",
        # "doc2vec",
        # "doc2vec_chunk",
        # "psif",
        # "bert_pt",
        # "bert_sentence_based_1000_pt",
        # "roberta_pt",
        # "roberta_sentence_based_1000_pt",
        # "xlm_pt",
        # "xlm_sentence_based_1000_pt",
        # "doc2vec_chunk",
        # "book2vec",
        # "book2vec_avg",
        # "book2vec_net_only_concat",
        # "book2vec_concat",
        # "book2vec_pca",
        # "book2vec_auto",
        # "book2vec_o_raw",
        # "book2vec_o_time",
        # "book2vec_o_plot",
        # "book2vec_o_loc",
        # "book2vec_o_atm",
        # "book2vec_o_sty",
        # "book2vec_wo_raw_concat",
        # "book2vec_net_concat",
        # "book2vec_dbow",
        # "book2vec_dbow_avg",
        # "book2vec_dbow_concat",
        # "book2vec_dbow_pca",
        # "book2vec_dbow_auto",
        "bert_sentence_based_1000_pt",
        "roberta_sentence_based_1000_pt",
        "xlm_sentence_based_1000_pt",


        # "book2vec_dbow_concat",
        # "bookwvec_window"
        # "book2vec_adv",
        # "book2vec_adv_concat",
    ]
    # EvaluationUtils.build_corpora()
    # EvaluationUtils.train_vecs()

    success_prediction_task("goodreads_genres", c.success_dict, algorithms)

    # genre_dict = {doc_id: document.genres for doc_id, document in c.documents.items()}
    #
    # genre_names = set([genre for doc_id, genre in genre_dict.items()])
    # genre_replace = {genre: i for i, genre in enumerate(genre_names)}
    # genre_dict = {doc_id: genre_replace[genre] for doc_id, genre in genre_dict.items()}
    #
    # genre_prediction_task("goodreads_genres", genre_dict, algorithms)

    df = pd.read_csv("../results/book_success_prediction/eval_scores.csv", index_col="Algorithm")

    # df = df.pivot(index='Algorithm', columns=['Classifier'], values='F1')
    print(df)
    print(df.columns)
    df = df[["NuSVC"]]
    df = df.round(4)
    print(df)
    print(df.to_latex())

