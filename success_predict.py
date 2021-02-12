from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from corpus_structure import Corpus, DataHandler
from vectorization import Vectorizer
from vectorization_utils import Vectorization
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd


def success_prediction_task(data_set_name: str, success_dict, vector_names):
    result_tuples = []
    for vectorization_algorithm in vector_names:
        # # vectorization_algorithm = "book2vec"
        # # corpus = Corpus.fast_load("all",
        # #                           "no_limit",
        # #                           data_set_name,
        # #                           "no_filter",
        # #                           "real",
        # #                           load_entities=False
        # #                           )
        # combine = None
        # vec_suffix = None
        # if "_concat" in vectorization_algorithm:
        #     vectorization_algorithm = vectorization_algorithm.replace("_concat", "")
        #     combine = "con"
        #     vec_suffix = "concat"
        # if "_pca" in vectorization_algorithm:
        #     vectorization_algorithm = vectorization_algorithm.replace("_pca", "")
        #     combine = "pca"
        #     vec_suffix = combine
        #
        # vec_path = Vectorization.build_vec_file_name("all",
        #                                              "no_limit",
        #                                              data_set_name,
        #                                              "no_filter",
        #                                              vectorization_algorithm,
        #                                              "real",
        #                                              allow_combination=True)
        # if combine:
        #     try:
        #         vectors = Vectorization.my_load_doc2vec_format(f'{vec_path}_{combine}')
        #     except FileNotFoundError:
        #         vectors = Vectorization.my_load_doc2vec_format(vec_path)
        #         doc_dict = {doctag: vectors.docvecs[doctag]
        #                     for doctag in vectors.docvecs.doctags if not str(doctag)[-1].isdigit()}
        #         # print(doc_dict)
        #         concat_vecs = Vectorization.combine_vectors_by_concat(doc_dict)
        #         vectors = Vectorization.store_vecs_and_reload(save_path=f'{vec_path}_{combine}', docs_dict=concat_vecs,
        #                                                       words_dict=None,
        #                                                       return_vecs=True)
        #     vectorization_algorithm = f'{vectorization_algorithm}_{vec_suffix}'
        #     # vectors = Vectorization.my_load_doc2vec_format(f'{vec_path}_con')
        # else:
        #     vectors = Vectorization.my_load_doc2vec_format(vec_path)

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
                y.append(0 if success_dict[doctag.replace(f"_{summation_method}", "")] == "failure" else 1)

            elif str(doctag)[-1].isdigit():
                x.append(vectors.docvecs[doctag])
                y.append(0 if success_dict[doctag] == "failure" else 1)
            else:
                pass

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
                                      f1_score(y_test, y_pred),
                                      precision_score(y_test, y_pred),
                                      recall_score(y_test, y_pred),
                                      accuracy_score(y_test, y_pred)))
    df = pd.DataFrame(result_tuples, columns=["Classifier", "Algorithm", "F1", "Precision", "Recall", "Accuracy"])
    df.to_csv("results/book_success_prediction/eval_scores.csv", index=False)
    print(df)


if __name__ == '__main__':
    c = DataHandler.load_maharjan_goodreads()
    algorithms = [
        "avg_wv2doc",
        "doc2vec",
        # "doc2vec_chunk",
        "book2vec",
        "book2vec_concat",
        "book2vec_pca",
        "book2vec_auto",
        "book2vec_o_raw",
        "book2vec_o_time",
        "book2vec_o_loc",
        "book2vec_o_atm",
        "book2vec_o_sty",
        "book2vec_adv",
        "book2vec_adv_concat",
    ]
    success_prediction_task("goodreads_genres", c.success_dict, algorithms)

