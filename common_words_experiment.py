# RealSeriesEvaluationRun.build_corpora()
# RealSeriesEvaluationRun.train_vecs()
import json
import os
from collections import defaultdict
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from corpus_structure import Corpus, DataHandler, ConfigLoader, Preprocesser, CommonWords
# from corpus_processing import Preprocesser, CommonWords
import matplotlib.pyplot as plt
import seaborn as sns


class CommonWordsExperiment:
    data_sets = [
        # "summaries",
        "german_books",
        # "german_series",
                 ]
    config = ConfigLoader.get_config()
    # filters = ["common_words_doc_freq"]
    thresholds = [
        0.00,
        0.005,
        0.01,
        0.015,
        0.0175,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06, 0.07, 0.08,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30, 0.35, 0.40, 0.45,
        0.50,
        0.55,
        0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
        0.90, 0.95,
        1.00
    ]
    absolute = True
    num_cores = 4

    @classmethod
    def filter_thresholds(cls, dir_path: str, parallel: bool = False):
        data_set_bar = tqdm(cls.data_sets, total=len(cls.data_sets), desc="2 Operate on dataset!!")
        for data_set in data_set_bar:
            data_set_bar.set_description(f'2 Operate on dataset >{data_set}<')
            data_set_bar.refresh()
            annotated_corpus_path = os.path.join(cls.config["system_storage"]["corpora"], data_set)
            try:
                corpus = Corpus.fast_load(path=annotated_corpus_path, load_entities=False)
            except FileNotFoundError:
                corpus = DataHandler.load_corpus(data_set)
                print('corpus loaded')
                # corpus = Preprocesser.annotate_corpus(corpus, without_spacy=False)
                # corpus.save_corpus_adv(annotated_corpus_path)
                Preprocesser.annotate_and_save(corpus,  corpus_dir=annotated_corpus_path, without_spacy=False)
                print('annotated corpus')
                del corpus
                corpus = Corpus.fast_load(path=annotated_corpus_path, load_entities=False)

                # print('saved corpus')

            if cls.absolute:
                thresholds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                              25, 50, 100, 1000, len(corpus)]
            else:
                thresholds = cls.thresholds

            threshold_bar = tqdm(thresholds, total=len(thresholds), desc="3 Calculate filter_mode results")
            if parallel:
                Parallel(n_jobs=cls.num_cores)(
                    delayed(CommonWordsExperiment.calculate_vocab_sizes)(corpus, t, data_set=data_set,
                                                                         dir_path=dir_path)
                    for t in threshold_bar)
            else:
                res = {t: CommonWordsExperiment.calculate_vocab_sizes(corpus, t, data_set=data_set,
                                                                      dir_path=dir_path)
                       for t in threshold_bar}

                with open(os.path.join(dir_path, 'all.json'), 'w', encoding='utf-8') as outfile:
                    json.dump(res, outfile, indent=1)

    @classmethod
    def calculate_vocab_sizes(cls, corpus: Corpus, threshold, data_set: str, dir_path: str):
        filtered_corpus_dir = Corpus.build_corpus_dir("",
                                                      "",
                                                      data_set,
                                                      f'specific_words_{threshold}',
                                                      "None").replace('__', '_')

        # print(os.path.isfile(os.path.join(dir_path.replace('.txt', ''), f'{threshold}.json')))

        if os.path.isfile(os.path.join(dir_path.replace('.txt', ''), f'{threshold}.json')):
            with open(os.path.join(dir_path.replace('.txt', ''), f'{threshold}.json'), 'r', encoding='utf-8') as file:
                data = json.load(file)
                print(threshold, data['global_vocab_size'])
                return data
        print('>|0', threshold)
        if not os.path.isdir(filtered_corpus_dir):
            if cls.absolute:
                to_specfic_words = CommonWords.global_too_specific_words_doc_frequency(
                    corpus, percentage_share=threshold, absolute_share=threshold)
            else:
                to_specfic_words = CommonWords.global_too_specific_words_doc_frequency(
                    corpus,
                    percentage_share=threshold)
            print('>|1 with len', len(to_specfic_words))
            # filtered_corpus = corpus.common_words_corpus_copy(to_specfic_words, masking=False)

            filtered_corpus = corpus.common_words_corpus_copy_mem_eff(to_specfic_words, masking=False,
                                                                      corpus_dir=filtered_corpus_dir,
                                                                      through_no_sentences_error=False)
        else:
            filtered_corpus = Corpus.load_corpus_from_dir_format(filtered_corpus_dir)
        # corpus.common_words_corpus_filtered(to_specfic_words, masking=False)
        # filtered_corpus = corpus
        # del corpus
        print('>|2')

        corpus_vocab_size = len(filtered_corpus.get_corpus_vocab())
        print('>|3 vocab size', corpus_vocab_size)
        document_sizes = {document_id:  {'vocab_size': document.vocab_size,
                                         'document_length': document.length}
                          for document_id, document in tqdm(filtered_corpus.documents.items(),
                                                            total=len(filtered_corpus),
                                                            desc="Calculate Corpus Sizes")}
        # for document_id, document in filtered_corpus.documents.items():
        #     print([token for token in document.get_flat_document_tokens() if token != 'del'][:100])

        # for document_id, words in common_words.items():
        #     print(document_id, len(words), document_sizes[document_id]['vocab_size'])
        vocab_sizes = []
        document_lengths = []
        for doc_id, document_size in document_sizes.items():
            vocab_sizes.append(document_size['vocab_size'])
            document_lengths.append(document_size['document_length'])

        print(threshold, corpus_vocab_size, np.mean(vocab_sizes), np.mean(document_lengths))

        result_dict = {'global_vocab_size': corpus_vocab_size,
                       'avg_vocab_size': np.mean(vocab_sizes),
                       'std_vocab_size': np.std(vocab_sizes),
                       'avg_document_length': np.mean(document_lengths),
                       'std_document_length': np.std(document_lengths),
                       'document_sizes': document_sizes}

        with open(os.path.join(dir_path, f'{threshold}.json'), 'w', encoding='utf-8') as outfile:
            json.dump(result_dict, outfile, indent=1)

        # print(filtered_corpus.get_corpus_vocab())
        # print(filtered_corpus.get_flat_document_tokens())
        return result_dict


def heatmap(matrix, title, years, cmap="YlGnBu", norm=True, diagonal_set=None):
    # normed_matrix = (matrix-matrix.mean())/matrix.std()
    if diagonal_set is not None:
        np.fill_diagonal(matrix, diagonal_set)
    if norm:
        matrix = matrix / matrix.max()

    sns.heatmap(matrix, cmap=cmap, xticklabels=years, yticklabels=years, square=True)
    plt.title(title, fontsize=20)
    plt.show()


def plot_single(x, y, title):
    plt.plot(x, y)
    plt.show()
    if title:
        plt.title(title)
    plt.xticks(rotation=90)
    plt.show()


def plot_many(x, y_list, title=None, skip=0, labels=None, plot_range=None, loc="best", n=None):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    plots = [ax.plot(x[skip:], y[skip:])[0] for y in y_list]
    # lgd = ax.legend(['Lag ' + str(lag) for lag in x], loc='center right', bbox_to_anchor=(1.3, 0.5))
    if labels:
        lgd = ax.legend(plots, labels, labelspacing=0., bbox_to_anchor=(1.04, 0.5), borderaxespad=0, loc=loc)
        # lgd = ax.legend(['Lag ' + str(lag) for lag in x], loc='center right', bbox_to_anchor=(1.3, 0.5))
    else:
        lgd = None

    if plot_range:
        plt.ylim(plot_range[0], plot_range[1])

    # plots = [plt.plot(x[skip:], y[skip:])[0] for y in y_list]
    # if labels:
    #     plt.legend(plots, labels, labelspacing=0., bbox_to_anchor=(1.04, 0.5), borderaxespad=0, loc="best")
    #
    # if range:
    #     plt.ylim(range[0], range[1])
    if title:
        plt.title(title)
    # #plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.xticks(rotation=90)
    plt.show()
    fig.savefig(f'type_ratio_{n}.png', dpi=600, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    # plt.savefig("output.png", bbox_inches="tight")


def violion_plot(dataframe):
    # todo different scales for absolute values
    sns.set_theme(style="whitegrid")
    _ = sns.violinplot(x="Threshold", y="Value", hue="Value Type",
                         data=dataframe, palette="muted", split=False)
    # ax2 = ax1.twinx()
    _ = sns.lineplot(data=dataframe, x="Threshold", y="Global Vocab Size")
    plt.show()


def lin_scale(x):
    return x


def plot_results(path: str):
    all_path = os.path.join(path, 'all.json')
    if os.path.isfile(all_path):
        with open(all_path, 'r', encoding='utf-8') as file:
            result = json.load(file)
    else:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        result = {}
        for file_path in files:
            with open(os.path.join(path, file_path), 'r', encoding='utf-8') as file:
                result[file_path.replace('.json', '')] = json.load(file)

        with open(os.path.join(path, 'all.json'), 'w', encoding='utf-8') as outfile:
            json.dump(result, outfile, indent=1)

    threshold_vals = []
    global_vocab_vals = []
    avg_vocab_size_vals = []
    avg_length_vals = []
    # "global_vocab_size": 319556,
    # "avg_vocab_size": 12851.589041095891,
    # "std_vocab_size": 5099.042552715211,
    # "avg_document_length": 107681.47945205479,
    # "std_document_length": 62174.930541816575,
    # "document_sizes": {
    orig_doc_vocab_len = None
    orig_doc_len = None
    orig_global_vocab_len = None
    diff_vocab_vals = []
    diff_doc_len_vals = []
    diff_global_vocab_vals = []
    scale = lin_scale  # np.log
    documents_vocab = defaultdict(list)
    documents_len = defaultdict(list)
    origin_doc_vocab = {}
    origin_doc_len = {}
    df_tuples = []
    for threshold, data in result.items():
        threshold_vals.append(float(threshold))

        if not orig_doc_len:
            orig_doc_vocab_len = data['avg_vocab_size']
            orig_doc_len = data['avg_document_length']
            orig_global_vocab_len = data['global_vocab_size']
            relative_vocab = 1
            relative_doc_len = 1
            relative_global_vocab = 1

        else:
            relative_vocab = 1 - scale(orig_doc_vocab_len - data['avg_vocab_size']) / orig_doc_vocab_len
            relative_doc_len = 1 - scale(orig_doc_len - data['avg_document_length']) / orig_doc_len
            relative_global_vocab = 1 - scale(orig_global_vocab_len - data['global_vocab_size']) / orig_global_vocab_len

        diff_vocab_vals.append(relative_vocab)
        diff_doc_len_vals.append(relative_doc_len)
        diff_global_vocab_vals.append(relative_global_vocab)

        global_vocab_vals.append(scale(data['global_vocab_size']))
        avg_vocab_size_vals.append(scale(data['avg_vocab_size']))
        avg_length_vals.append(scale(data['avg_document_length']))

        for doc_id, vals in data['document_sizes'].items():
            documents_vocab[doc_id].append(vals['vocab_size'])
            documents_len[doc_id].append(vals['document_length'])
            # print(origin_doc_vocab)
            if doc_id not in origin_doc_vocab:
                origin_doc_vocab[doc_id] = vals['vocab_size']
                origin_doc_len[doc_id] = vals['document_length']
                relative_vocab = 1
                relative_doc_len = 1

            else:
                relative_vocab = 1 - (origin_doc_vocab[doc_id] - vals['vocab_size']) / origin_doc_vocab[doc_id]
                relative_doc_len = 1 - (origin_doc_len[doc_id] - vals['document_length']) / origin_doc_len[doc_id]

            df_tuples.append((doc_id, threshold, relative_vocab, 'Vocabulary Size Loss', relative_global_vocab))
            df_tuples.append((doc_id, threshold, relative_doc_len, 'Document Length Loss', relative_global_vocab))

    df = pd.DataFrame(df_tuples, columns=['Document ID', 'Threshold', 'Value', 'Value Type', 'Global Vocab Size'])
    violion_plot(df)

    documents_vocab_ls = [val for _, val in documents_vocab.items()]
    documents_len_ls = [val for _, val in documents_len.items()]

    b = []
    b.extend(documents_vocab_ls)
    b.extend(documents_len_ls)

    plot_many(threshold_vals,
              [diff_global_vocab_vals,
               # avg_vocab_size_vals,
               # avg_length_vals,
               diff_vocab_vals,
               diff_doc_len_vals
               ],
              labels=['Global Vocab Size',
                      # 'Avg Vocab Size',
                      # 'Avg Document Length',
                      'Diff Vocab',
                      'Diff Doc Len'
                      ], title=None)

    # plot_many(threshold_vals,
    #           [diff_vocab_vals,
    #            diff_doc_len_vals
    #            ],
    #           labels=[
    #                   'Diff Vocab',
    #                   'Diff Doc Len'
    #                   ], title=None)

    # plot_many(threshold_vals, b, title=None)

    # print(threshold_vals)
    # plt.plot(threshold_vals, global_vocab_vals)
    # plt.show()


if __name__ == '__main__':
    path_name = 'results/common_words_experiment/threshold_values'
    CommonWordsExperiment.filter_thresholds(path_name)

    plot_results(path_name)
