from collections import defaultdict

from scipy.stats import stats

from corpus_structure import Corpus
from vectorization import Vectorizer
import pandas as pd

import matplotlib.pyplot as plt
from math import pi


def radar_chart(df):
    # number of variable
    categories = list(df)
    len_vars = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(len_vars) * 2 * pi for n in range(len_vars)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=7)
    plt.ylim(0.0, 1.0)

    # Ind1
    values = df.iloc[0].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.index[0])
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    try:
        values = df.iloc[1].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=df.index[1])
        ax.fill(angles, values, 'r', alpha=0.1)
    except (KeyError, IndexError):
        pass

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.show()


def get_facet_sims_of_books(vectors, doc_id_a: str, doc_id_b: str):
    tuples = Vectorizer.get_facet_sims(vectors, doc_id_a, doc_id_b)
    df = pd.DataFrame(tuples, columns=["Facet", "ID_A", "ID_B", "Similarity"])
    df = df.pivot(index=["ID_A", "ID_B"], columns="Facet", values="Similarity")
    return df


def loop_facets(vectors, corpus: Corpus):
    tuples = []
    for i, doc_id_a in enumerate(corpus.documents.keys()):
        for j, doc_id_b in enumerate(corpus.documents.keys()):
            if j > i:
                tuples.extend(Vectorizer.get_facet_sims(vectors, doc_id_a, doc_id_b))
    df = pd.DataFrame(tuples, columns=["Facet", "ID_A", "ID_B", "Similarity"])
    return df


def correlation(human_df, vectors):
    all_values = defaultdict(list)
    facet_human_vals = defaultdict(list)
    facet_pred_vals = defaultdict(list)

    for i, row in human_df.iterrows():
        pred_df = get_facet_sims_of_books(vectors, row['ID_A'], row['ID_B'])

        real_val = row["Similarity"]
        pred_val = pred_df[row["Facet"]].values[0]

        facet_human_vals[row["Facet"]].append(real_val)
        facet_pred_vals[row["Facet"]].append(pred_val)

        all_values['real'].append(real_val)
        all_values['predicted'].append(pred_val)

    # noinspection PyTypeChecker
    complete_correlation = stats.spearmanr(all_values['real'], all_values['predicted'])

    facet_correlation = {}
    for facet in facet_human_vals:
        # noinspection PyTypeChecker
        facet_correlation[facet] = stats.spearmanr(facet_human_vals[facet], facet_pred_vals[facet])

    return complete_correlation, facet_correlation


if __name__ == '__main__':
    c = Corpus.fast_load(path="corpora/german_series", load_entities=False)

    vec_path = Vectorizer.build_vec_file_name("all",
                                              "no_limit",
                                              "german_series",
                                              "no_filter",
                                              "book2vec",
                                              "real")

    vecs = Vectorizer.my_load_doc2vec_format(vec_path)

    big_df = loop_facets(vecs, c)

    fake_df = pd.DataFrame([('gs_0_0', 'gs_0_1', 'time', 0.9), ('gs_0_0', 'gs_0_1', 'sty', 0.91),
                            ('gs_0_0', 'gs_10_0', 'time', 0.4), ('gs_0_0', 'gs_10_0', 'sty', 0.5),
                            ('gs_0_0', 'gs_1_0', 'time', 0.6), ('gs_0_0', 'gs_1_0', 'sty', 0.3)],
                           columns=['ID_A', 'ID_B', 'Facet', 'Similarity'])
    complete_corr, facet_cors = correlation(fake_df, vecs)
    print(complete_corr, facet_cors)

    radar_chart(get_facet_sims_of_books(vecs, c[0].doc_id, c[1].doc_id))
    radar_chart(get_facet_sims_of_books(vecs, c[0].doc_id, c[2].doc_id))
    radar_chart(get_facet_sims_of_books(vecs, c[0].doc_id, c[3].doc_id))
    radar_chart(get_facet_sims_of_books(vecs, c[0].doc_id, c[10].doc_id))

    # get_facet_sims_of_books(vecs, corpus[0].doc_id, corpus[2].doc_id)
    # get_facet_sims_of_books(vecs, corpus[0].doc_id, corpus[3].doc_id)
    # get_facet_sims_of_books(vecs, corpus[0].doc_id, corpus[10].doc_id)
