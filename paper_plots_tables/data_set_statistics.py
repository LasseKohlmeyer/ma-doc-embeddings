from collections import defaultdict
from typing import List

from lib2vec.corpus_structure import Corpus, Language
import numpy as np
import pandas as pd
from scipy.stats import iqr
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def corpus_stats(data_sets: List[str]):
    tuples = []

    for data_set_name in data_sets:
        corpus = Corpus.fast_load("all",
                                  "no_limit",
                                  data_set_name,
                                  "no_filter",
                                  "real",
                                  load_entities=False
                                  )
        if corpus.language == Language.DE:
            language = "GER"
        else:
            language = "EN"
        nr_books = human_format(len(corpus.documents))

        document_tokens = [document.length for document in corpus.documents.values()]
        tokens_total = human_format(sum(document_tokens))
        tokens_avg = f'{np.mean(document_tokens):.0f} ± {np.std(document_tokens):.0f}'
        # tokens_median = f'{np.median(document_tokens):.0f} ± {iqr(document_tokens):.0f}'
        tokens_median = f'{human_format(np.median(document_tokens))}'
        tokens_iqr = f'{human_format(iqr(document_tokens))}'
        tokens_min = f'{human_format(np.min(document_tokens))}'
        tokens_max = f'{human_format(np.max(document_tokens))}'
        document_vocab = [document.vocab_size for document in corpus.documents.values()]
        vocab_total = human_format(sum(document_vocab))
        vocab_avg = f'{np.mean(document_vocab):.0f} ± {np.std(document_vocab):.0f}'
        # vocab_median = f'{np.median(document_vocab):.0f} ± {iqr(document_vocab):.0f}'
        vocab_median = f'{human_format(np.median(document_vocab))}'
        vocab_iqr = f'{human_format(iqr(document_vocab))}'
        # vocab_mix = f'[{human_format(np.min(document_vocab))}, {human_format(np.max(document_vocab))}]'
        vocab_min = f'{human_format(np.min(document_vocab))}'
        vocab_max = f'{human_format(np.max(document_vocab))}'

        document_sents = [document.sentences_nr for document in corpus.documents.values()]
        sents_total = sum(document_sents)
        sents_avg = f'{np.mean(document_sents):.0f} ± {np.std(document_sents):.0f}'
        sents_median = f'{np.median(document_sents):.0f} ± {iqr(document_sents):.0f}'

        author_dict = defaultdict(list)
        for doc_id, document in corpus.documents.items():
            author_dict[document.authors].append(doc_id)

        print({author: len(doc_ids) for author, doc_ids in author_dict.items() if author is not None})
        author_vals = [len(doc_ids) for author, doc_ids in author_dict.items() if author is not None]

        author_median = f'{np.median(author_vals):.0f} ± {iqr(author_vals):.0f} [{np.min(author_vals):.0f}, {np.max(author_vals):.0f}]'
        # author_mean = f'{np.mean(author_vals):.2f} ± {np.std(author_vals):.2f} [{np.min(author_vals):.0f}, {np.max(author_vals):.0f}]'
        author_mean = f'{np.mean(author_vals):.2f}'
        author_std = f'{np.std(author_vals):.2f}'
        author_mix = f'[{np.min(author_vals):.0f}, {np.max(author_vals):.0f}]'
        author_max = f'{np.max(author_vals):.0f}'

        print(data_set_name, "Author median iqr / mean std", author_median, author_mean)
        if corpus.series_dict and len(corpus.series_dict) > 0:
            series_vals = [len(doc_ids) for series_id, doc_ids in corpus.series_dict.items() if series_id is not None]
            series_median = f'{np.median(series_vals):.0f} ± {iqr(series_vals):.0f} [{np.min(series_vals):.0f}, {np.max(series_vals):.0f}]'
            # series_mean = f'{np.mean(series_vals):.2f} ± {np.std(series_vals):.2f} [{np.min(series_vals):.0f}, {np.max(series_vals):.0f}]'

            series_mean = f'{np.mean(series_vals):.2f}'
            series_std = f'{np.std(series_vals):.2f}'
            series_mix = f'[{np.min(series_vals):.0f}, {np.max(series_vals):.0f}]'

            series_max = f'{np.max(series_vals):.0f}'
            print(data_set_name, "Series median iqr / mean std", series_median, series_mean)
        else:
            series_median = "-"
            series_mean = "-"
            series_std = "-"
            series_mix = "-"

        if corpus.shared_attributes_dict is None:
            corpus.calculate_documents_with_shared_attributes()
        if corpus.shared_attributes_dict["same_genres"] and len(corpus.shared_attributes_dict["same_genres"]) > 1:
            genre_vals = [len(doc_ids) for genre, doc_ids in corpus.shared_attributes_dict["same_genres"].items() if genre is not None]
            # print(genre_vals)
            genre_median = f'{np.median(genre_vals):.0f} ± {iqr(genre_vals):.0f} [{np.min(genre_vals):.0f}, {np.max(genre_vals):.0f}]'
            # genre_mean = f'{np.mean(genre_vals):.2f} ± {np.std(genre_vals):.2f} [{np.min(genre_vals):.0f}, {np.max(genre_vals):.0f}]'
            genre_mean = f'{np.mean(genre_vals):.2f}'
            genre_std = f'{np.std(genre_vals):.2f}'
            genre_mix = f'[{np.min(genre_vals):.0f}, {np.max(genre_vals):.0f}]'

            print(data_set_name, "Genre median iqr / mean std", genre_median, genre_mean)
        else:
            genre_median = "-"
            genre_mean = "-"
            genre_std = "-"
            genre_mix = "-"



        # if corpus and len(corpus.series_dict) > 0:
        #     series_median = np.median([len(doc_ids) for series_id, doc_ids in corpus.series_dict.items()])

        tuples.append((data_set_name,
                       nr_books, language,
                       tokens_total, tokens_median, tokens_iqr, tokens_min, tokens_max,
                       vocab_total, vocab_median, vocab_iqr, vocab_min, vocab_max,
                       author_mean, author_std, author_mix,
                       series_mean, series_std, series_mix,
                       genre_mean, genre_std, genre_mix,))
    df = pd.DataFrame(tuples, columns=["Data set", "Amount of Books", "Language",
                                       "Total Tokens", "Tokens Median", "Tokens IQR",
                                       "Tokens Min", "Tokens Max",
                                       "Total Vocabulary", "Vocabulary Median",  "Vocabulary IQR",
                                       "Vocabulary Min", "Vocabulary Max",
                                       "Author Mean", "Author STD", "Author [Min, Max]",
                                       "Series Mean", "Series STD", "Series [Min, Max]",
                                       "Genre Mean", "Genre STD", "Genre [Min, Max]"
                                       # "Books by Same Author ± STD [Min, Max]",
                                       # "Books by Same Series ± STD [Min, Max]",
                                       # "Books by Same Genre ± STD [Min, Max]",
                                       # "Total Sentences", "Sentences Mean [STD]", "Sentences Median [IQR]",
                                       ], index=data_sets)
    df = df.transpose()
    print(df)
    df.to_csv("results/dataset_stats/sizes.csv", index=True)
    print(df.to_latex(index=True))


if __name__ == '__main__':
    data_sets = [
        "german_books",
        "german_series",
        "dta",
        "dta_series",
        "litrec",
        "goodreads_genres",
        "classic_gutenberg",
        ]

    corpus_stats(data_sets)

