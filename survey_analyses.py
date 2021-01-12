from collections import defaultdict
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa


def fleiss_kappa_self(M):
    """Computes Fleiss' kappa for group of annotators.
    :param M: a matrix of shape (:attr:'N', :attr:'k') with 'N' = number of subjects (items)
    and 'k' = the number of categories (ratings).
        'M[i, j]' represent the number of raters who assigned the 'i'th subject to the 'j'th category.
    :type: numpy matrix
    :rtype: float
    :return: Fleiss' kappa score
    """
    N, k = M.shape  # N is # of items, k is # of categories
    n_annotators = float(np.sum(M[0, :]))  # # of annotators

    tot_annotations = N * n_annotators  # the total # of annotations
    # print(n_annotators, tot_annotations)
    category_sum = np.sum(M, axis=0)  # the sum of each category over all items

    # chance agreement
    p = category_sum / tot_annotations  # the distribution of each category over all annotations
    PbarE = np.sum(p * p)  # average chance agreement over all categories

    # observed agreement
    P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
    Pbar = np.sum(P) / N  # add all observed agreement chances per item and divide by amount of items

    return round((Pbar - PbarE) / (1 - PbarE), 4)


def annswer_to_kappa_helper(answers_as_list, values):
    unique, counts = np.unique(answers_as_list, return_counts=True)
    count_dict = dict(zip(unique, counts))
    ratings = []
    for value in values:
        if value in count_dict:
            ratings.append(count_dict[value])
        else:
            ratings.append(0)
    return ratings


def answers_to_kappa_matrix(answers_as_list):
    values = [0.0, 0.25, 0.45, 0.5, 0.55, 0.75, 1]

    unique, counts = np.unique(answers_as_list, return_counts=True)
    count_dict = dict(zip(unique, counts))
    ratings = annswer_to_kappa_helper(answers_as_list, values)

    kappa_matrix = np.array(ratings).reshape(1, len(values))
    # print(kappa_matrix)
    return kappa_matrix


def kappa_matrix_of_answer_dict(answ_dict: Dict[Tuple[str, str], List[float]]):
    values = [0.0, 0.25, 0.45, 0.5, 0.55, 0.75, 1]
    kappa_matrix = []
    for entry, answs in answ_dict.items():
        answs = [answer for answer in answs if answer > 0]
        ratings = annswer_to_kappa_helper(answs, values)

        # print('55', answs, ratings)

        kappa_matrix.append(np.array(ratings))
    kappa_matrix = np.array(kappa_matrix)
    # print('55', answer_dict, kappa_matrix)
    return kappa_matrix


if __name__ == "__main__":

    df = pd.read_csv("data_websci_2021-01-07_14-32.csv", delimiter='\t', encoding="utf-16")
    df = df.fillna(-10)

    comparison_suffix_mapping = {
        "01": "time",
        "02": "location",
        "03": "content",
        "04": "plot",
        "05": "atmosphere",
        "06": "total",
    }

    books_answer_mapping = {
        1: "unknown",
        2: "known, but not read",
        3: "read before 10 years",
        4: "read in last 10 years",
        -9: "unknown",
        -10: "unknown"
    }

    comparison_answer_mapping = {
        1: 0.0,
        2.0: 0.25,
        3.0: 0.5,
        4.0: 0.75,
        5: 1.0,
        -1: 0.45,
        -2: 0.55,
        -9: -1.0,
        -10: -1.0,
    }

    books_mapping = [['A Doll\'s House by Henrik Ibsen', 'Nora oder Ein Puppenheim von Henrik Ibsen',
                      'CP03_01'],
                     ['A Tale of Two Cities by Charles Dickens', 'Eine Geschichte aus zwei Städten von Charles Dickens',
                      'CP03_02'],
                     ['Adventures of Huckleberry Finn by Mark Twain', 'Die Abenteuer des Huckleberry Finn von Mark Twain',
                      'CP03_03'],
                     ['Alice’s Adventures in Wonderland by Lewis Carroll', 'Alice im Wunderland von Lewis Carroll',
                      'CP03_04'],
                     ['Dracula by Bram Stoker', 'Dracula von Bram Stoker',
                      'CP03_05'],
                     ['Emma by Jane Austen', 'Emma von Jane Austen', ''
                                                                     'CP03_06'],
                     ['Frankenstein by Mary Shelley', 'Frankenstein; oder: Der moderne Prometheus von Mary Shelley',
                      'CP03_07'],
                     ['Great Expectations by Charles Dickens', 'Große Erwartungen von Charles Dickens',
                      'CP03_08'],
                     ['Metamorphosis by Franz Kafka', 'Die Verwandlung von Franz Kafka',
                      'CP03_09'],
                     ['Pride and Prejudice by Jane Austen', 'Stolz und Vorurteil von Jane Austen',
                      'CP03_10'],
                     ['The Adventures of Sherlock Holmes Arthur C. Doyle',
                      'Die Abenteuer des Sherlock Holmes von Arthur C. Doyle',
                      'CP03_11'],
                     ['The Adventures of Tom Sawyer by Mark Twain', 'Die Abenteuer des Tom Sawyer von Mark Twain',
                      'CP03_12'],
                     ['The Count of Monte Cristo by Alexandre Dumas', 'Der Graf von Monte Christo von Alexandre Dumas',
                      'CP03_13'],
                     ['The Picture of Dorian Gray by Oscar Wilde', 'Das Bildnis des Dorian Gray von Oscar Wilde',
                      'CP03_14'],
                     ['The Yellow Wallpaper by Charlotte P. Gilman', 'Die gelbe Tapete von Charlotte P. Gilman',
                      'CP03_15'],
                     ['Heart of Darkness by Joseph Conrad', 'Herz der Finsternis von Joseph Conrad',
                      'CP03_16'],
                     ['Moby Dick by Herman Melville', 'Moby-Dick; oder: Der Wal von Herman Melville',
                      'CP03_17'],
                     ['War and Peace by Leo Tolstoy', 'Krieg und Frieden von Leo Tolstoy',
                      'CP03_18'],
                     ['Wuthering Heights by Emily Brontë', 'Sturmhöhe von Emily Brontë',
                      'CP03_19'],
                     ['Treasure Island by Robert L. Stevenson', 'Die Schatzinsel von Robert L. Stevenson',
                      'CP03_20']]

    book_id2_comp = {
        "01": {"02": "CG01", "03": "CG02", "06": "CG03", "07": "CG04", "09": "CG05", "10": "CG06", "14": "CG07",
               "15": "CG08", "19": "CG09", "20": "CG10"},
        "02": {"01": "CG01", "04": "CG11", "05": "CG12", "09": "CG13", "11": "CG14", "12": "CG15", "13": "CG16",
               "16": "CG17", "18": "CG18", "19": "CG19"},
        "03": {"01": "CG02", "04": "CG20", "08": "CG21", "09": "CG22", "10": "CG23", "12": "CG24", "16": "CG25",
               "17": "CG26", "19": "CG27", "20": "CG28"},
        "04": {"02": "CG11", "03": "CG20", "05": "CG29", "06": "CG30", "07": "CG31", "09": "CG32", "11": "CG33",
               "14": "CG34", "15": "CG35", "17": "CG36"},
        "05": {"02": "CG12", "04": "CG29", "06": "CG37", "07": "CG38", "09": "CG39", "11": "CG40", "12": "CG41",
               "13": "CG42", "14": "CG43", "16": "CG44"},
        "06": {"01": "CG03", "04": "CG30", "05": "CG37", "07": "CG45", "10": "CG46", "15": "CG47", "17": "CG48",
               "18": "CG49", "19": "CG50", "20": "CG51"},
        "07": {"01": "CG04", "04": "CG31", "05": "CG38", "06": "CG45", "09": "CG52", "11": "CG53", "13": "CG54",
               "14": "CG55", "15": "CG56", "17": "CG57"},
        "08": {"03": "CG21", "10": "CG58", "11": "CG59", "12": "CG60", "15": "CG61", "16": "CG62", "17": "CG63",
               "18": "CG64", "19": "CG65", "20": "CG66"},
        "09": {"01": "CG05", "02": "CG13", "03": "CG22", "04": "CG32", "05": "CG39", "07": "CG52", "14": "CG67",
               "16": "CG68", "20": "CG69"},
        "10": {"01": "CG06", "03": "CG23", "06": "CG46", "08": "CG58", "13": "CG70", "14": "CG71", "15": "CG72",
               "18": "CG73", "19": "CG74"},
        "11": {"02": "CG14", "04": "CG33", "05": "CG40", "07": "CG53", "08": "CG59", "12": "CG75", "13": "CG76",
               "14": "CG77", "17": "CG78", "20": "CG79"},
        "12": {"02": "CG15", "03": "CG24", "05": "CG41", "08": "CG60", "11": "CG75", "13": "CG80", "14": "CG81",
               "15": "CG82", "16": "CG83", "20": "CG84"},
        "13": {"02": "CG16", "05": "CG42", "07": "CG54", "10": "CG70", "11": "CG76", "12": "CG80", "18": "CG85",
               "19": "CG86"},
        "14": {"01": "CG07", "04": "CG34", "05": "CG43", "07": "CG55", "09": "CG67", "10": "CG71", "11": "CG77",
               "12": "CG81", "16": "CG87", "17": "CG88"},
        "15": {"01": "CG08", "04": "CG35", "06": "CG47", "07": "CG56", "08": "CG61", "10": "CG72", "12": "CG82",
               "18": "CG89", "19": "CG90", "20": "CG91"},
        "16": {"02": "CG17", "03": "CG25", "05": "CG44", "08": "CG62", "09": "CG68", "12": "CG83", "14": "CG87",
               "17": "CG92", "18": "CG93", "20": "CG94"},
        "17": {"03": "CG26", "04": "CG36", "06": "CG48", "07": "CG57", "08": "CG63", "11": "CG78", "14": "CG88",
               "16": "CG92", "18": "CG95", "20": "CG96"},
        "18": {"02": "CG18", "06": "CG49", "08": "CG64", "10": "CG73", "13": "CG85", "15": "CG89", "16": "CG93",
               "17": "CG95", "19": "CG97", "20": "CG98"},
        "19": {"01": "CG09", "02": "CG19", "03": "CG27", "06": "CG50", "08": "CG65", "10": "CG74", "13": "CG86",
               "15": "CG90", "18": "CG97", "20": "CG99"},
        "20": {"01": "CG10", "03": "CG28", "06": "CG51", "08": "CG66", "09": "CG69", "11": "CG79", "12": "CG84",
               "15": "CG91", "16": "CG94", "17": "CG96", "18": "CG98", "19": "CG99"}
    }
    comp_id2book_id = {}
    for key_a, a_keys in book_id2_comp.items():
        for key_b, comp_id in a_keys.items():
            comp_id2book_id[comp_id] = (key_b, key_a)
    # print(comp_id2book_id)
    book_id2english_title = {}
    for (english, german, index) in books_mapping:
        book_id2english_title[index.replace('CP03_', '')] = english

    # print(book_id2english_title)

    languages = df["LANGUAGE"].tolist()
    # Overview to Book knowledge with language sensitivity
    language_books_dict = defaultdict(lambda: defaultdict(dict))
    for column in df.columns:
        if str(column).startswith("CP03"):
            prefix, suffix = column.split('_')
            answers = [books_answer_mapping[answer] for answer in df[column].tolist()]

            language_dict = defaultdict(list)
            for language, answer in zip(languages, answers):
                language_dict[language].append(answer)
            language_dict["all"].extend(language_dict["ger"])
            language_dict["all"].extend(language_dict["eng"])

            counter_language_dict = {}
            for language, language_books in language_dict.items():
                counter_dict = {"unknown": 0, "known, but not read": 0,
                                "read before 10 years": 0, "read in last 10 years": 0}
                for answer in language_books:
                    counter_dict[answer] += 1

                counter_language_dict[language] = counter_dict

            for key in language_dict:
                # language_books_dict[key][suffix].extend(language_dict[key])
                language_books_dict[key][suffix] = counter_language_dict[key]

    # print(language_books_dict)

    book_tuples = []
    for language, language_books in language_books_dict.items():
        for book_id, book_answers in language_books.items():
            book_tuples.append((book_id, book_id2english_title[book_id], language, book_answers["unknown"], book_answers["known, but not read"],
                                book_answers["read before 10 years"], book_answers["read in last 10 years"]))
    book_familarity_df = pd.DataFrame(book_tuples, columns=["Book ID", "Description", "Language", "Unknown", "Known, but not read",
                                                            "Read before 10 years", "Read in last 10 years"])
    book_familarity_df = book_familarity_df.sort_values(by=['Book ID']).set_index(["Book ID", "Language"])
    # print(book_familarity_df)
    book_familarity_df.to_csv('results/human_assessment/book_familarity.csv')

    # print('-----------------------------------------------------')

    language_answer_dict = defaultdict(lambda: defaultdict(list))
    for column in df.columns:
        if str(column).startswith("CG"):
            prefix, suffix = column.split('_')
            answers = [comparison_answer_mapping[answer] for answer in df[column].tolist()]

            language_dict = defaultdict(list)
            for language, answer in zip(languages, answers):
                if answer >= 0:
                    language_dict[language].append(answer)
            language_dict["all"].extend(language_dict["ger"])
            language_dict["all"].extend(language_dict["eng"])
            for key in language_dict:
                language_answer_dict[key][(prefix, comparison_suffix_mapping[suffix])].extend(language_dict[key])
            # answer_dict[(prefix, comparison_suffix_mapping[suffix])].extend(language_dict)

    # print(answer_dict["all"])
    print(fleiss_kappa_self(kappa_matrix_of_answer_dict(language_answer_dict["all"])))
    print(fleiss_kappa_self(kappa_matrix_of_answer_dict(language_answer_dict["ger"])))
    # print(fleiss_kappa_self(kappa_matrix_of_answer_dict(answer_dict["eng"])))

    # aggregated values for each book pair and language
    tuples = []
    for language, answer_dict in language_answer_dict.items():
        # overal, german, english
        language_dict = defaultdict(list)

        for comparison, answers in answer_dict.items():
            if len(answers) == 0:
                mean = None
                std = None
                fleiss_kappa_score = None
            else:
                answers = np.array(answers)
                mean = answers.mean()
                std = answers.std()
                fleiss_kappa_score = fleiss_kappa_self(answers_to_kappa_matrix(answers))

            book_id_a = comp_id2book_id[comparison[0]][0]
            book_id_b = comp_id2book_id[comparison[0]][1]
            tuples.append((language, comparison[0], book_id_a, book_id2english_title[book_id_a], book_id_b,
                           book_id2english_title[book_id_b],
                           comparison[1], mean, std, fleiss_kappa_score))

    comparison_results_df = pd.DataFrame(tuples, columns=["Language", "Comparison ID",
                                                          "Book ID A", "Description Book A",
                                                          "Book ID B", "Description Book B",
                                                          "Similarity Type", "Mean",
                                                          "STD", "Fleiss Kappa"])

    # print(comparison_results_df)
    comparison_results_df.to_csv('results/human_assessment/book_comparison.csv', index=False)




