from collections import defaultdict
from typing import List

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 24})
import pandas as pd


def draw_result(lst_iter, list_of_lists, title, metric: str):
    for label, lst_loss in list_of_lists.items():
        plt.plot(lst_iter, lst_loss[0], lst_loss[1], label=label,  linewidth=4.0)
    # plt.plot(lst_iter, lst_acc, '-r', label='accuracy')

    plt.xlabel("Dimension Size")
    plt.ylabel(metric.upper())
    plt.legend(loc='best')
    if title:
        plt.title(title)

    # save image
    # plt.savefig(title+".png")  # should before show method

    # show
    plt.show()


def draw_result_multi_metrics(lst_iter, list_of_lists, title, metric: List[str]):
    for label, lst_loss in list_of_lists.items():
        plt.plot(lst_iter, lst_loss[0], lst_loss[1], label=label, linewidth=4.0, markersize=15)
        # plt.plot(lst_iter, lst_loss[0], f"{lst_loss[1][0]}-", label=label, linewidth=4.0)
    # plt.plot(lst_iter, lst_acc, '-r', label='accuracy')

    plt.xlabel("Window Size")
    plt.ylabel("Metric Score")
    plt.legend(loc='center right')


    if title:
        plt.title(title)

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    # save image
    # plt.savefig(title+".png")  # should before show method

    # show
    plt.show()


def draw_loss_curve_from_df(df: pd.DataFrame):
    # iteration num
    print(df.columns)
    algo_dict = defaultdict(list)
    dims = []
    metric = "ndcg"
    for i, row in df.iterrows():
        algorithm = row["Algorithm"]
        algorithm_parts = algorithm.split('_')
        if len(algorithm_parts) == 2:
            name = algorithm_parts[0]
            dim = algorithm_parts[1].replace('win', '')
        elif len(algorithm_parts) == 3:
            name = f'{algorithm_parts[0]}_{algorithm_parts[2]}'
            dim = algorithm_parts[1].replace('win', '')
        else:
            raise UserWarning("No proper format")
        if dim not in dims:
            dims.append(dim)
        algo_dict[name].append((row["Task"], dim, float(row[metric].split()[0])))

    print(algo_dict)

    lst_iter = dims

    # loss of iteration
    list_of_lists = {
        "doc2vec Author": ([val[2] for val in algo_dict["doc2vec"] if val[0] == "AuthorTask"], 'r-'),
        "doc2vec Series": ([val[2] for val in algo_dict["doc2vec"] if val[0] == "SeriesTask"], 'r--'),
        "lib2vec Author": ([val[2] for val in algo_dict["book2vec"] if val[0] == "AuthorTask"],'g-'),
        "lib2vec Series": ([val[2] for val in algo_dict["book2vec"] if val[0] == "SeriesTask"], 'g--'),
        "book2vec_concat Author": ([val[2] for val in algo_dict["book2vec_concat"] if val[0] == "AuthorTask"], 'b-'),
        "book2vec_concat Series": ([val[2] for val in algo_dict["book2vec_concat"] if val[0] == "SeriesTask"], 'b--'),
                     }
    # lst_loss = [val[2] for val in algo_dict["doc2vec"] if val[0] == "SeriesTask"]
    # lst_loss = np.random.randn(1, 100).reshape((100, ))

    draw_result(lst_iter, list_of_lists, None, metric)


def draw_multi_metric_curve_from_df(df: pd.DataFrame):
    # iteration num
    print(df.columns)
    algo_dict = defaultdict(list)
    windows = []
    metrics = ["ndcg", "prec10", "f_prec10", "rec10", "f110"]
    for i, row in df.iterrows():
        algorithm = row["Algorithm"]
        if "doc2vec" not in algorithm:
            continue
        algorithm_parts = algorithm.split('_')
        if len(algorithm_parts) == 2:
            name = algorithm_parts[0]
            win = int(algorithm_parts[1].replace('win', ''))
        elif len(algorithm_parts) == 3:
            name = f'{algorithm_parts[0]}_{algorithm_parts[2]}'
            win = int(algorithm_parts[1].replace('win', ''))
        else:
            raise UserWarning("No proper format")
        if win not in windows:
            windows.append(win)

        # metric_results = {}
        for metric in metrics:
            # metric_results[metric].append(float(row[metric].split()[0]))
            algo_dict[name].append((row["Task"], win, metric, float(row[metric].split()[0])))

        # algo_dict[name].append((row["Task"], dim, metric_results))

    print(algo_dict)

    lst_iter = windows

    # loss of iteration
    list_of_lists = {
        "Author NDCG": ([val[3] for val in algo_dict["doc2vec"]
                                 if val[0] == "AuthorTask" and val[2] == "ndcg"], '--r^'),

        "Series NDCG": ([val[3] for val in algo_dict["doc2vec"]
                         if val[0] == "SeriesTask" and val[2] == "ndcg"], '--ro'),

        # "Author FairP@10": ([val[3] for val in algo_dict["doc2vec"]
        #                          if val[0] == "AuthorTask" and val[2] == "f_prec10"], '--c^'),
        # "Series FairP@10": ([val[3] for val in algo_dict["doc2vec"]
        #                  if val[0] == "SeriesTask" and val[2] == "f_prec10"], '--co'),
        "Author P@10": ([val[3] for val in algo_dict["doc2vec"]
                         if val[0] == "AuthorTask" and val[2] == "prec10"], '--b^'),
        "Series P@10": ([val[3] for val in algo_dict["doc2vec"]
                         if val[0] == "SeriesTask" and val[2] == "prec10"], '--bo'),
        "Author R@10": ([val[3] for val in algo_dict["doc2vec"]
                         if val[0] == "AuthorTask" and val[2] == "rec10"], '--g^'),
        "Series R@10": ([val[3] for val in algo_dict["doc2vec"]
                         if val[0] == "SeriesTask" and val[2] == "rec10"], '--go'),
        "Author F1@10": ([val[3] for val in algo_dict["doc2vec"]
                                 if val[0] == "AuthorTask" and val[2] == "f110"], '--k^'),
        "Series F1@10": ([val[3] for val in algo_dict["doc2vec"]
                          if val[0] == "SeriesTask" and val[2] == "f110"], '--ko'),

        # "doc2vec Author AP": ([val[3] for val in algo_dict["doc2vec"]
        #                          if val[0] == "AuthorTask" and val[2] == "ap"], 'y-'),
        # "doc2vec Author MRR": ([val[3] for val in algo_dict["doc2vec"]
        #                          if val[0] == "AuthorTask" and val[2] == "mrr"], 'c-'),





        # "doc2vec Series AP": ([val[3] for val in algo_dict["doc2vec"]
        #                        if val[0] == "SeriesTask" and val[2] == "ap"], 'y--'),
        # "doc2vec Series MRR": ([val[3] for val in algo_dict["doc2vec"]
        #                         if val[0] == "SeriesTask" and val[2] == "mrr"], 'c--'),

        # "book2vec Series NDCG": ([val[3] for val in algo_dict["book2vec"]
        #                          if val[0] == "SeriesTask" and val[2] == "ndcg"], 'r-.'),
        # "book2vec Series P@10": ([val[3] for val in algo_dict["book2vec"]
        #                          if val[0] == "SeriesTask" and val[2] == "f_prec10"], 'g-.'),
        # "book2vec Series R@10": ([val[3] for val in algo_dict["book2vec"]
        #                          if val[0] == "SeriesTask" and val[2] == "rec10"], 'b-.'),
        # "book2vec Series F1@10": ([val[3] for val in algo_dict["book2vec"]
        #                           if val[0] == "SeriesTask" and val[2] == "f110"], 'k-.'),

        # "lib2vec Author": ([val[2] for val in algo_dict["book2vec"] if val[0] == "AuthorTask"],'g-'),
        # "lib2vec Series": ([val[2] for val in algo_dict["book2vec"] if val[0] == "SeriesTask"], 'g--'),
        # "book2vec_concat Author": ([val[2] for val in algo_dict["book2vec_concat"] if val[0] == "AuthorTask"], 'b-'),
        # "book2vec_concat Series": ([val[2] for val in algo_dict["book2vec_concat"] if val[0] == "SeriesTask"], 'b--'),
                     }
    # lst_loss = [val[2] for val in algo_dict["doc2vec"] if val[0] == "SeriesTask"]
    # lst_loss = np.random.randn(1, 100).reshape((100, ))
    print(list_of_lists)
    draw_result_multi_metrics(lst_iter, list_of_lists, None, metrics)


def test_draw():
    # iteration num
    lst_iter = range(100)

    # loss of iteration
    lst_loss = [0.01 * i + 0.01 * i ** 2 for i in range(100)]
    # lst_loss = np.random.randn(1, 100).reshape((100, ))

    # accuracy of iteration
    lst_acc = [0.01 * i - 0.01 * i ** 2 for i in range(100)]
    # lst_acc = np.random.randn(1, 100).reshape((100, ))
    draw_result(lst_iter, lst_loss, "sgd_method")


if __name__ == '__main__':
    # df = pd.read_csv("result_doc_window/z_table_gb.csv")
    # print(df.columns)
    # plot_column = "ndcg"
    # draw_loss_curve_from_df(pd.read_csv("result_doc_window/z_table_gb.csv"))
    draw_multi_metric_curve_from_df(pd.read_csv("../result_doc_window/z_table.csv"))

    # test_draw()