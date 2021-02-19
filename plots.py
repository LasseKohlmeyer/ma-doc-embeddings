import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
import pandas as pd

tips = sns.load_dataset("tips")
import random
import numpy as np


def grouped_barplot(df: pd.DataFrame, x, sub_group, y, err):
    u = df[x].unique()
    x = np.arange(len(u))
    subx = df[sub_group].unique()
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
    width = np.diff(offsets).mean()
    for i, gr in enumerate(subx):
        dfg = df[df[sub_group] == gr]
        plt.bar(x + offsets[i], dfg[y].values, width=width,
                label="{} {}".format(sub_group, gr), yerr=dfg[err].values)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(x, u)
    plt.legend()
    plt.show()


def facet_barplot():
    base_df = pd.read_csv("results/z_table.csv")
    print(base_df)
    facets = ["time", "loc", "sty", "atm", "raw", "cont", "plot"]
    with_without = ["with", "with_out"]
    tuples = []
    for facet in facets:
        for w in with_without:
            tuples.append((random.uniform(0.2, 1), random.uniform(0.0, 0.15), facet, w))

    # tuples = [(0.9, "with", "loc"), (0.8, "without", "loc"), (0.95, "with", "time"), (0.5, "without", "time")]
    df = pd.DataFrame(tuples, columns=["Score", "STD", "Facet", "Status"])
    print(df)

    grouped_barplot(df, x="Facet", sub_group="Status", y="Score", err="STD")


facet_barplot()
