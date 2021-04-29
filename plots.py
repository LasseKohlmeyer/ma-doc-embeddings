import random

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# plt.rcParams.update({'font.size': 18})
sns.set_theme(style="whitegrid", font_scale=1.8)



def show_values_on_bars(axs):
    def _show_on_single_plot(ax):
        for p in ax.patches:
            _x = p.get_x() + p.get_width() / 2
            _y = p.get_y() + p.get_height()
            value = '{:.3f}'.format(p.get_height())
            ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)

def grouped_barplot(df: pd.DataFrame, x, sub_group, y, y_label, err, line, task):
    x_label = x
    u = df[x].unique()
    x = np.arange(len(u))
    subx = df[sub_group].unique()
    offsets = (np.arange(len(subx)) - np.arange(len(subx)).mean()) / (len(subx) + 1.)
    width = np.diff(offsets).mean()
    fig, ax = plt.subplots(1, 1)
    for i, gr in enumerate(subx):
        dfg = df[df[sub_group] == gr]
        plt.bar(x + offsets[i], dfg[y].values, width=width,
                label="{}".format(gr), yerr=dfg[err].values)
    plt.axhline(y=line, color='r', linestyle='-')
    show_values_on_bars(ax)
    plt.xlabel(x_label)
    if y_label == "prec10":
        y_label = "Prec@10"
    else:
        y_label = y_label.upper()

    plt.ylabel(f"{task.replace('Task', '')} {y_label}")
    plt.xticks(x, u)
    plt.legend(loc="lower right")
    plt.ylim([0, 1.0])
    plt.grid(False)
    plt.show()


facet_map = {"time": "Time", "loc": "Location", "sty": "Style", "atm": "Atmosphere", "raw": "Raw", "cont": "Content",
             "plot": "Plot"}


def facet_barplot(input_path: str = "results/z_table_gb.csv",
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
    total_val = algorithm_dict["book2vec_concat"]
    for algorithm, (score, std) in algorithm_dict.items():
        for facet in facets:
            if facet in algorithm:
                if "_o_" in algorithm:
                    tuples.append((score, std, facet_map[facet], "Only Facet"))
                elif "_wo_" in algorithm:
                    tuples.append((score, std, facet_map[facet], "Without Facet"))
                else:
                    raise UserWarning("not identified facet!")

    df = pd.DataFrame(tuples, columns=["Score", "STD", "Facet", "Status"])
    print(df)

    # with_without = ["with", "with_out"]
    # tuples = []
    # for facet in facets:
    #     for w in with_without:
    #         tuples.append((random.uniform(0.2, 1), random.uniform(0.0, 0.15), facet, w))

    # tuples = [(0.9, "with", "loc"), (0.8, "without", "loc"), (0.95, "with", "time"), (0.5, "without", "time")]

    df = pd.DataFrame(tuples, columns=["Score", "STD", "Facet", "Status"])
    print(df)

    grouped_barplot(df, x="Facet", sub_group="Status", y="Score", y_label=metric, err="STD", line=total_val[0], task=task)


facet_barplot(input_path="result_series_facets/z_table_comb.csv",
              series_length="all",
              filter="specific_words_strict",
              datasset="litrec",
              task="AuthorTask",
              metric="ndcg")
