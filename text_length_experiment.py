import multiprocessing
import os
from typing import Dict, Union

from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm
from corpus_structure import Corpus, ConfigLoader
from vectorization import Vectorizer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from sklearn.preprocessing import minmax_scale

from vectorization_utils import Vectorization


def get_short_mid_long(df, column):
    q1_of_length = int(df[[column]].quantile(q=0.333333333))
    q3_of_length = int(df[[column]].quantile(q=0.666666666))
    # print(q1_of_length, q3_of_length)
    filter_1q = df[df[column] <= q1_of_length]
    filter_2q = df[(q1_of_length < df[column]) & (df[column] <= q3_of_length)]
    filter_3q = df[q3_of_length < df[column]]
    return filter_1q, filter_2q, filter_3q


class HistScatter:
    def __init__(self, df, x0_label, x1_label, algorithm_name):
        x = df[[x0_label, x1_label]].to_numpy()
        y_full = df[['Length A']].to_numpy().flatten()

        self.df = df
        self.title = algorithm_name
        self.cmap = getattr(cm, 'plasma_r', cm.hot_r)
        self.x0_label = x0_label
        self.x1_label = x1_label
        self.x = x
        self.y_full = y_full
        self.y = minmax_scale(y_full)
        self.distributions = [
            (f'', self.x),
        ]

        self.make_plot(0)

    @staticmethod
    def create_axes(title, figsize=(8, 6)):
        fig = plt.figure(figsize=figsize)
        fig.suptitle(title)

        # define the axis for the first plot
        left, width = 0.1, 0.55
        bottom, height = 0.1, 0.7
        bottom_h = height + 0.15
        left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.1]
        rect_histy = [left_h, bottom, 0.05, height]

        ax_scatter = plt.axes(rect_scatter)
        ax_histx = plt.axes(rect_histx)
        ax_histy = plt.axes(rect_histy)

        # define the axis for the zoomed-in plot
        # left = width + left + 0.2
        # left_h = left + width + 0.02

        # rect_scatter = [left, bottom, width, height]
        # rect_histx = [left, bottom_h, width, 0.1]
        # rect_histy = [left_h, bottom, 0.05, height]
        #
        # ax_scatter_zoom = plt.axes(rect_scatter)
        # ax_histx_zoom = plt.axes(rect_histx)
        # ax_histy_zoom = plt.axes(rect_histy)

        # define the axis for the colorbar
        left, width = width + left + 0.13, 0.01

        rect_colorbar = [left, bottom, width, height]
        ax_colorbar = plt.axes(rect_colorbar)

        return ((ax_scatter, ax_histy, ax_histx),
                # (ax_scatter_zoom, ax_histy_zoom, ax_histx_zoom),
                ax_colorbar)

    def plot_distribution(self, axes, x, y, hist_nbins=50, title="",
                          x0_label="", x1_label=""):
        ax, hist_x1, hist_x0 = axes

        ax.set_title(title)
        ax.set_xlabel(x0_label)
        ax.set_ylabel(x1_label)

        # The scatter plot
        colors = self.cmap(y)
        ax.scatter(x[:, 0], x[:, 1], alpha=0.5, marker='o', s=5, lw=0, c=colors)

        filter_1q, filter_2q, filter_3q = get_short_mid_long(self.df, "Length A")

        z = filter_1q[[self.x0_label, self.x1_label]].to_numpy()
        m, b = np.polyfit(x=z[:, 0], y=z[:, 1], deg=1)
        q1_line, = ax.plot(z[:, 0], m * z[:, 0] + b, 'g--')
        q1_line.set_label('Length Smaller as the 1st Quartile')

        z = filter_2q[[self.x0_label, self.x1_label]].to_numpy()
        m, b = np.polyfit(x=z[:, 0], y=z[:, 1], deg=1)
        q2_line, = ax.plot(z[:, 0], m * z[:, 0] + b, 'r--')
        q2_line.set_label('Length Between 1st and 3rd Quartile')

        for d_id in (sorted(set(filter_3q["Doc ID A"].to_numpy().flatten()))):
            print(d_id)
        z = filter_3q[[self.x0_label, self.x1_label]].to_numpy()
        m, b = np.polyfit(x=z[:, 0], y=z[:, 1], deg=1)
        q3_line, = ax.plot(z[:, 0], m * z[:, 0] + b, 'b--')
        q3_line.set_label('Length Bigger as the 3rd Quartile')

        ax.legend()
        # Removing the top and the right spine for aesthetics
        # make nice axis layout
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.spines['left'].set_position(('outward', 10))
        ax.spines['bottom'].set_position(('outward', 10))

        # Histogram for axis X1 (feature 5)
        hist_x1.set_ylim(ax.get_ylim())
        hist_x1.hist(x[:, 1], bins=hist_nbins, orientation='horizontal',
                     color='grey', ec='grey')
        hist_x1.axis('off')

        # Histogram for axis X0 (feature 0)
        hist_x0.set_xlim(ax.get_xlim())
        hist_x0.hist(x[:, 0], bins=hist_nbins, orientation='vertical',
                     color='grey', ec='grey')
        hist_x0.axis('off')

    def make_plot(self, item_idx):
        title, x = self.distributions[item_idx]
        # ax_zoom_out, ax_zoom_in, ax_colorbar = self.create_axes(title)
        ax_zoom_out, ax_colorbar = self.create_axes(title)
        axarr = ax_zoom_out
        self.plot_distribution(axarr, x, self.y, hist_nbins=100,
                               x0_label=self.x0_label,
                               x1_label=self.x1_label)

        # zoom-in
        # zoom_in_percentile_range = (0, 99)
        # cutoffs_X0 = np.percentile(x[:, 0], zoom_in_percentile_range)
        # cutoffs_X1 = np.percentile(x[:, 1], zoom_in_percentile_range)

        # non_outliers_mask = (
        #     np.all(x > [cutoffs_X0[0], cutoffs_X1[0]], axis=1) &
        #     np.all(x < [cutoffs_X0[1], cutoffs_X1[1]], axis=1))
        # self.plot_distribution(axarr[1], x[non_outliers_mask], self.y[non_outliers_mask],
        #                        hist_nbins=50,
        #                        x0_label=self.x0_label,
        #                        x1_label=self.x1_label,
        #                        title="Zoom-in")

        norm = mpl.colors.Normalize(self.y_full.min(), self.y_full.max())
        mpl.colorbar.ColorbarBase(ax_colorbar, cmap=self.cmap,
                                  norm=norm, orientation='vertical',
                                  label='Absolute Text Length')
        # plt.show()
        plt.savefig(fname=os.path.join('plots', f'length_influence_{self.title}.png'), dpi=600)


def histogram(data: Dict[str, Union[int, float]]):
    tuples = []
    for idx, val in data.items():
        tuples.append((idx, val))

    df = pd.DataFrame(tuples, columns=['Document ID', 'Length'])
    print(df)
    sns.histplot(data=df, x="Length", bins=50)
    plt.show()


class TextLengthExperiment:
    config = ConfigLoader.get_config()
    num_cores = int(0.75 * multiprocessing.cpu_count())

    ignore_same = True

    data_sets = [
        "german_series",
        # "dta_series"
    ]
    filters = [
        "no_filter",
        # "named_entities",
        # "common_words_strict",
        # "common_words_strict_general_words_sensitive",
        # "common_words_relaxed",
        # "common_words_relaxed_general_words_sensitive",
        # "common_words_doc_freq"
        # "stopwords",
        # "nouns",
        # "verbs",
        # "adjectives",
        # "avn"
    ]
    vectorization_algorithms = [
        "avg_wv2doc",
        "doc2vec",
        # "longformer_untuned"
        "book2vec",
        # "book2vec_o_raw",
        # "book2vec_o_loc",
        # "book2vec_o_time",
        # "book2vec_o_sty",
        # "book2vec_o_atm",
        # "book2vec_wo_raw",
        # "book2vec_wo_loc",
        # "book2vec_wo_time",
        # "book2vec_wo_sty",
        # "book2vec_wo_atm",
        # "book2vec_w2v",
        "book2vec_adv",
        # "book2vec_adv_o_raw",
        # "book2vec_adv_o_loc",
        # "book2vec_adv_o_time",
        # "book2vec_adv_o_sty",
        # "book2vec_adv_o_atm",
        # "book2vec_adv_o_plot",
        # "book2vec_adv_o_cont",
        # "book2vec_adv_wo_raw",
        # "book2vec_adv_wo_loc",
        # "book2vec_adv_wo_time",
        # "book2vec_adv_wo_sty",
        # "book2vec_adv_wo_atm",
        # "book2vec_adv_wo_plot",
        # "book2vec_adv_wo_cont",
        # "random_aspect2vec"
        # "avg_wv2doc_untrained",
        # "doc2vec_untrained",
        # "book2vec_untrained",
    ]

    @classmethod
    def run_experiment(cls, parallel: bool = False):
        # res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))
        for data_set in tqdm(cls.data_sets, total=len(cls.data_sets), desc=f"Evaluate datasets"):
            for filter_mode in tqdm(cls.filters, total=len(cls.filters), desc=f"Evaluate filters"):
                corpus = Corpus.fast_load("all",
                                          "no_limit",
                                          data_set,
                                          filter_mode,
                                          "real",
                                          load_entities=False
                                          )

                vec_bar = tqdm(cls.vectorization_algorithms,
                               total=len(cls.vectorization_algorithms),
                               desc=f"Evaluate algorithm")
                if parallel:
                    tuple_list_results = Parallel(n_jobs=cls.num_cores)(
                        delayed(TextLengthExperiment.eval_vec_loop_eff)(corpus,
                                                                        "all",
                                                                        "no_limit",
                                                                        data_set,
                                                                        filter_mode,
                                                                        vectorization_algorithm)
                        for vectorization_algorithm in vec_bar)
                else:
                    tuple_list_results = [TextLengthExperiment.eval_vec_loop_eff(corpus,
                                                                                 "all",
                                                                                 "no_limit",
                                                                                 data_set,
                                                                                 filter_mode,
                                                                                 vectorization_algorithm)
                                          for vectorization_algorithm in vec_bar]

                full_df = pd.DataFrame(tuple_list_results, columns=['Algorithm',
                                                                    'Full Spearman [p]',
                                                                    'Short Spearman [p]',
                                                                    'Medium Spearman [p]',
                                                                    'Long Spearman [p]'])

                full_df.to_csv(os.path.join('results', 'text_length_experiment', 'text_length_spearman.csv'),
                               index=False)
                full_df.to_latex(os.path.join('results', 'text_length_experiment', 'text_length_spearman.tex'),
                                 index=False)

                # for subpart_nr, data, filt_mod, vec_algo, results in tuple_list_results:
                #     res[subpart_nr][data][filt_mod][vec_algo] = results

    @staticmethod
    def length_similarity(length_lookup, doc_id_a, doc_id_b):
        a = length_lookup[doc_id_a]
        b = length_lookup[doc_id_b]
        return min(a, b) / max(a, b)

    @staticmethod
    def length_abs(length_lookup, doc_id_a, doc_id_b):
        a = length_lookup[doc_id_a]
        b = length_lookup[doc_id_b]
        return abs(a - b)

    @staticmethod
    def combined_sim(vectors, length_lookup, doc_id_a, doc_id_b):
        return vectors.docvecs.similarity(doc_id_a, doc_id_b), TextLengthExperiment.length_similarity(length_lookup,
                                                                                                      doc_id_a,
                                                                                                      doc_id_b)

    @staticmethod
    def modified_doc_id(doc_id, modificator):
        if modificator is "NF":
            return doc_id
        return f'{doc_id}_{modificator}'

    @staticmethod
    def triangle_values(input_df: pd.DataFrame, matrix_column: str) -> np.ndarray:
        cosine_matrix_df = input_df
        columns_to_drop = ['Cosine Similarity', 'Length Similarity', 'Length Distance', 'Length A', 'Length B']
        columns_to_drop.remove(matrix_column)
        cosine_matrix_df = cosine_matrix_df.drop(columns_to_drop, axis=1)
        cosine_matrix_df = cosine_matrix_df.set_index(['Doc ID A'])
        # cosine_matrix_df["Doc ID A"] = pd.Categorical(cosine_matrix_df["Doc ID A"],
        #                                               categories=cosine_matrix_df["Doc ID A"].unique(), ordered=True)
        # cosine_matrix_df["Doc ID B"] = pd.Categorical(cosine_matrix_df["Doc ID B"],
        #                                               categories=cosine_matrix_df["Doc ID B"].unique(), ordered=True)
        cosine_matrix_df = cosine_matrix_df.pivot(columns='Doc ID B')[matrix_column]
        tria = cosine_matrix_df.mask(np.triu(np.ones(cosine_matrix_df.shape)).astype(bool)).stack().to_numpy().flatten()
        return tria

    @classmethod
    def eval_vec_loop_eff(cls, corpus, number_of_subparts, corpus_size, data_set, filter_mode, vectorization_algorithm):
        vec_path = Vectorization.build_vec_file_name(number_of_subparts,
                                                     corpus_size,
                                                     data_set,
                                                     filter_mode,
                                                     vectorization_algorithm,
                                                     "real")
        summation_method = "NF"
        try:
            vectors = Vectorization.my_load_doc2vec_format(vec_path)
        except FileNotFoundError:
            if "_o_" in vectorization_algorithm:
                vec_splitted = vectorization_algorithm.split("_o_")[0]
                focus_facette = vectorization_algorithm.split("_o_")[1]
                base_algorithm = vec_splitted
                vec_path = Vectorization.build_vec_file_name(number_of_subparts,
                                                             corpus_size,
                                                             data_set,
                                                             filter_mode,
                                                             base_algorithm,
                                                             "real")
                vectors = Vectorization.my_load_doc2vec_format(vec_path)
                summation_method = focus_facette
            else:
                raise FileNotFoundError

        doctags = vectors.docvecs.doctags.keys()
        doctags = [doctag for doctag in doctags if doctag[-1].isdigit()]
        length_vals = {doc_id: len(document.get_flat_tokens_from_disk()) for doc_id, document in
                       corpus.documents.items()}
        # length_vals = {doc_id: len(document.get_flat_document_tokens()) for doc_id, document in
        #                corpus.documents.items()}
        # print(length_vals)
        # histogram(length_vals)

        full_tuples = []
        for doc_id_a in doctags:
            for doc_id_b in doctags:
                cos_sim = vectors.docvecs.similarity(cls.modified_doc_id(doc_id_a, summation_method),
                                                     cls.modified_doc_id(doc_id_b, summation_method))
                if cos_sim < 0:
                    cos_sim = -1 * cos_sim
                length_sim = TextLengthExperiment.length_similarity(length_vals, doc_id_a, doc_id_b)
                length_abs = TextLengthExperiment.length_abs(length_vals, doc_id_a, doc_id_b)
                full_tuples.append((doc_id_a, doc_id_b,
                                    cos_sim, length_sim, length_abs,
                                    length_vals[doc_id_a], length_vals[doc_id_b])
                                   )
        full_df = pd.DataFrame(full_tuples, columns=['Doc ID A', 'Doc ID B',
                                                     'Cosine Similarity', 'Length Similarity', 'Length Distance',
                                                     'Length A', 'Length B'])
        print(full_df)

        tria_cos = cls.triangle_values(full_df, 'Cosine Similarity')
        # tria_len_d = cls.triangle_values(full_df, 'Length Distance')
        tria_len = cls.triangle_values(full_df, 'Length Similarity')

        tuples = []
        for cosine, length, length_a in zip(full_df['Cosine Similarity'].to_numpy().flatten(),
                                            full_df['Length Similarity'].to_numpy().flatten(),
                                            full_df['Length A'].to_numpy().flatten()):
            tuples.append((cosine, length, length_a))

        # df = pd.DataFrame(tuples, columns=['Cosine Similarity', 'Length Similarity', 'Length A'])
        # pd.plotting.scatter_matrix(df, hist_kwds={'bins': 50})
        # plt.show()

        HistScatter(full_df, x0_label='Cosine Similarity', x1_label='Length Similarity',
                    algorithm_name=vectorization_algorithm)

        # print(vectorization_algorithm, stats.pearsonr(full_df[['Cosine Similarity']].to_numpy().flatten(),
        #                                               full_df[['Length Simimarity']].to_numpy().flatten()))

        # print(vectorization_algorithm, stats.pearsonr(tria_cos, tria_len))

        filter_1q, filter_2q, filter_3q = get_short_mid_long(full_df, "Length A")

        # noinspection PyTypeChecker
        full_spearman = stats.spearmanr(tria_cos, tria_len)

        full_spearman = f'{full_spearman[0]:.4f} [{full_spearman[1]:.4f}]'

        short_spearman = stats.spearmanr(filter_1q[['Cosine Similarity']].to_numpy().flatten(),
                                         filter_1q[['Length Similarity']].to_numpy().flatten())

        short_spearman = f'{short_spearman[0]:.4f} [{short_spearman[1]:.4f}]'

        mid_spearman = stats.spearmanr(filter_2q[['Cosine Similarity']].to_numpy().flatten(),
                                       filter_2q[['Length Similarity']].to_numpy().flatten())

        mid_spearman = f'{mid_spearman[0]:.4f} [{mid_spearman[1]:.4f}]'

        long_spearman = stats.spearmanr(filter_3q[['Cosine Similarity']].to_numpy().flatten(),
                                        filter_3q[['Length Similarity']].to_numpy().flatten())

        long_spearman = f'{long_spearman[0]:.4f} [{long_spearman[1]:.4f}]'

        # print(vectorization_algorithm, stats.pearsonr(tria_cos, tria_len_d))
        # print(vectorization_algorithm, stats.spearmanr(tria_cos, tria_len_d))
        # cosine_df = pd.DataFrame(cosine_matrix, index=doctags, columns=doctags)

        return vectorization_algorithm, full_spearman, short_spearman, mid_spearman, long_spearman


if __name__ == '__main__':
    # dataset = fetch_california_housing()
    # X_full, y_full = dataset.data, dataset.target
    # x = X_full[:, [0, 5]]
    # print(x)
    # print(y_full)
    # df = pd.DataFrame([(1,2,1),(2,3,0),(3,4,1),(4,5,0),(1,4,1),(4,3,0)],
    #                   columns=['Cosine Similarity', 'Length Similarity', 'Length A'])
    # HistScatter(df, x0_label='Length Similarity', x1_label='Cosine Similarity', algorithm_name="No")
    # histogram({'d1': 2, 'd3': 3, 'd2': 2, 'd4': 5})
    # TextLengthExperiment.run_experiment()
    c = Corpus.fast_load("all",
                         "no_limit",
                         "german_series",
                         "no_filter",
                         "real",
                         load_entities=False
                         )
    length_values = {doc_id: document.vocab_size for doc_id, document in
                     c.documents.items()}
    histogram(length_values)

# avg_wv2doc SpearmanrResult(correlation=0.1822316899913066, pvalue=1.03923709619589e-79)
# doc2vec SpearmanrResult(correlation=0.07783613065249204, pvalue=1.0637541425586075e-15)
# book2vec SpearmanrResult(correlation=0.05916974075425099, pvalue=1.1136615858540168e-09)
