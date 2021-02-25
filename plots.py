import random

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")


def grouped_barplot(df: pd.DataFrame, x, sub_group, y, y_label, err):
    x_label = x
    u = df[x].unique()
    x = np.arange(len(u))
    subx = df[sub_group].unique()
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
    width = np.diff(offsets).mean()
    for i, gr in enumerate(subx):
        dfg = df[df[sub_group] == gr]
        plt.bar(x + offsets[i], dfg[y].values, width=width,
                label="{} {}".format(sub_group, gr), yerr=dfg[err].values)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x, u)
    plt.legend()
    plt.show()


def facet_barplot(input_path: str = "results/z_table.csv",
                  series_length: str = "all",
                  filter: str = "no_filter",
                  datasset: str = "german_books",
                  task: str = "AuthorTask",
                  metric: str = "ndcg"):
    facets = ["time", "loc", "sty", "atm", "raw", "cont", "plot"]

    base_df = pd.read_csv(input_path)
    algorithm_dict = {}
    for i, row in base_df.iterrows():
        score, std = [float(s) for s in str(row[metric]).split(" ± ")]
        if row["Series_length"] == series_length and row["Dataset"] == datasset and row["Task"] == task \
                and row["Filter"] == filter:
            algorithm_dict[row["Algorithm"]] = (score, std)

    tuples = []
    print(algorithm_dict)
    for algorithm, (score, std) in algorithm_dict.items():
        for facet in facets:
            if facet in algorithm:
                if "_o_" in algorithm:
                    tuples.append((score, std, facet, "with"))
                elif "_wo_" in algorithm:
                    tuples.append((score, std, facet, "with_out"))
                else:
                    raise UserWarning("not identified facet!")

    # df = pd.DataFrame(tuples, columns=["Score", "STD", "Facet", "Status"])
    # print(df)

    with_without = ["with", "with_out"]
    tuples = []
    for facet in facets:
        for w in with_without:
            tuples.append((random.uniform(0.2, 1), random.uniform(0.0, 0.15), facet, w))

    # tuples = [(0.9, "with", "loc"), (0.8, "without", "loc"), (0.95, "with", "time"), (0.5, "without", "time")]

    df = pd.DataFrame(tuples, columns=["Score", "STD", "Facet", "Status"])
    print(df)

    grouped_barplot(df, x="Facet", sub_group="Status", y="Score", y_label=metric, err="STD")


facet_barplot(input_path="results/z_table.csv",
              series_length="all",
              filter="no_filter",
              datasset="german_books",
              task="AuthorTask",
              metric="ndcg")
