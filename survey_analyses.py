from collections import defaultdict, Counter
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import scipy.stats
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


def generate_facet_column_names(facet_name:str):
    def increment_str(inp):
        if inp == "01":
            return "02"
        if inp == "03":
            return "04"
        if inp == "05":
            return "06"
        if inp == "07":
            return "08"
        if inp == "09":
            return "10"

    base_suffixes = ["01", "03", "05", "07", "09"]
    res = {}
    for base_suffix in base_suffixes:
        res[f'{facet_name}{base_suffix}'] = (f'{facet_name}{increment_str(base_suffix)}_01',
                                             f'{facet_name}{increment_str(base_suffix)}_02',
                                             f'{facet_name}{increment_str(base_suffix)}_03')
    return res


def selection_map(book1: str, book2: str, book3: str, selection):
    if selection == -10 or selection == -10.0 or selection == "-10" or selection == "-10.0" or \
            selection == -9 or selection == -9.0 or selection == "-9" or selection == "-9.0":
        return "skip"
    elif selection == 1 or selection == 1.0 or selection == "1" or selection == "-1.0":
        return f'1|{book1}=={book2}'
    elif selection == 2 or selection == 2.0 or selection == "2" or selection == "-2.0":
        return f'2|{book1}=={book3}'
    elif selection == 3 or selection == 3.0 or selection == "3" or selection == "-3.0":
        return f'3|{book2}=={book3}'
    else:
        raise UserWarning(f"No matches for {selection}!")


def category_ranks(rating):
    if '|' in rating:
        return int(rating.split('|')[0])
    else:
        return 0


def kappa_score_single(list_of_ratings: List[str]):
    category_counter = {0: 0, 1: 0, 2: 10, 3: 0}

    mapped_ranks = [category_ranks(rating) for rating in list_of_ratings]

    for category in mapped_ranks:
        category_counter[category] += 1
    frequencies = [categopry_frequency for categopry_frequency in category_counter.values()]
    matrix = np.array(frequencies).reshape(1, len(frequencies))

    for frequency in frequencies:
        if sum(frequencies) == frequency:
            return 1.0, frequencies
    # print(matrix, matrix.shape)
    return fleiss_kappa_self(matrix), frequencies


def majority_vote(list_of_ratings: List[str]):
    if len(set(list_of_ratings)) < 2:
        return list_of_ratings[0]
    counter = Counter(list_of_ratings)
    most_common = counter.most_common(2)
    if most_common[0][1] == most_common[1][1]:
        return "unsure"
    else:
        return most_common[0][0]


def group_kappa_for_df(tri_df: pd.DataFrame, kappa_column: str = None):
    tuple_dict = defaultdict(list)
    voted_dict = {}
    if kappa_column is None:
        for i, row in tri_df.iterrows():
            tuple_dict[(row["Book 1"],
                        row["Book 2"],
                        row["Book 3"],
                        row["Facet"])].append(row["Selection"])

        kappa_multi = defaultdict(list)
        for (book_1, book_2, book3, facet_name), values in tuple_dict.items():
            # print(key, len(values))
            kappa_s, frequencies = kappa_score_single(values)
            kappa_multi["all"].append(np.array(frequencies))
            voted = majority_vote(values)
            # print(kappa_s)
            voted_dict[(book_1, book_2, book3, facet_name)] = voted
    else:
        for i, row in tri_df.iterrows():
            tuple_dict[(row["Book 1"],
                        row["Book 2"],
                        row["Book 3"],
                        row["Facet"],
                        row[kappa_column])].append(row["Selection"])

        kappa_multi = defaultdict(list)
        for (book_1, book_2, book3, facet_name, col), values in tuple_dict.items():
            # print(key, len(values))
            kappa_s, frequencies = kappa_score_single(values)
            kappa_multi[col].append(np.array(frequencies))
            voted = majority_vote(values)
            # print(kappa_s)
            voted_dict[(book_1, book_2, book3, facet_name)] = voted
    d = {}
    for kappa_column_key, kappa_values in kappa_multi.items():
        kappa_multi = np.array(kappa_values)
        d[kappa_column_key] = fleiss_kappa_self(kappa_multi)

    return d, voted_dict


def facet_kappa_for_df(tri_df: pd.DataFrame):
    tuple_dict = defaultdict(list)
    for i, row in tri_df.iterrows():
        tuple_dict[(row["Book 1"],
                    row["Book 2"],
                    row["Book 3"],
                    row["Facet"])].append(row["Selection"])

    kappa_multi = defaultdict(list)
    for (book_1, book_2, book3, facet_name), values in tuple_dict.items():
        # print(key, len(values))
        kappa_s, frequencies = kappa_score_single(values)
        kappa_multi[facet_name].append(np.array(frequencies))

    d = {}
    for kappa_column_key, kappa_values in kappa_multi.items():
        kappa_multi = np.array(kappa_values)
        d[kappa_column_key] = fleiss_kappa_self(kappa_multi)

    return d


if __name__ == "__main__":

    df = pd.read_csv("data_websci_2021-01-15_10-57.csv", delimiter='\t', encoding="utf-16")
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
        2: "known",
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

    books_mapping = [["Uncle Tom's Cabin by Harriet Beecher Stowe", "Onkel Toms Hütte von Harriet Beecher Stowe",
                      'CP08_01'],
                     ['A Tale of Two Cities by Charles Dickens', 'Eine Geschichte aus zwei Städten von Charles Dickens',
                      'CP08_02'],
                     ['Adventures of Huckleberry Finn by Mark Twain', 'Die Abenteuer des Huckleberry Finn von Mark Twain',
                      'CP08_03'],
                     ['Alice’s Adventures in Wonderland by Lewis Carroll', 'Alice im Wunderland von Lewis Carroll',
                      'CP08_04'],
                     ['Dracula by Bram Stoker', 'Dracula von Bram Stoker',
                      'CP08_05'],
                     ['Emma by Jane Austen', 'Emma von Jane Austen',
                      'CP08_06'],
                     ['Frankenstein by Mary Shelley', 'Frankenstein; oder: Der moderne Prometheus von Mary Shelley',
                      'CP08_07'],
                     ['Great Expectations by Charles Dickens', 'Große Erwartungen von Charles Dickens',
                      'CP08_08'],
                     ['Metamorphosis by Franz Kafka', 'Die Verwandlung von Franz Kafka',
                      'CP08_09'],
                     ['Pride and Prejudice by Jane Austen', 'Stolz und Vorurteil von Jane Austen',
                      'CP08_10'],
                     ['The Adventures of Sherlock Holmes Arthur C. Doyle',
                      'Die Abenteuer des Sherlock Holmes von Arthur C. Doyle',
                      'CP08_11'],
                     ['The Adventures of Tom Sawyer by Mark Twain', 'Die Abenteuer des Tom Sawyer von Mark Twain',
                      'CP08_12'],
                     ['The Count of Monte Cristo by Alexandre Dumas', 'Der Graf von Monte Christo von Alexandre Dumas',
                      'CP08_13'],
                     ['The Picture of Dorian Gray by Oscar Wilde', 'Das Bildnis des Dorian Gray von Oscar Wilde',
                      'CP08_14'],
                     ['Little Women by Louisa M. Alcott', 'Little Women von Louisa M. Alcott',
                      'CP08_15'],
                     ['Heart of Darkness by Joseph Conrad', 'Herz der Finsternis von Joseph Conrad',
                      'CP08_16'],
                     ['Moby Dick by Herman Melville', 'Moby-Dick; oder: Der Wal von Herman Melville',
                      'CP08_17'],
                     ['War and Peace by Leo Tolstoy', 'Krieg und Frieden von Leo Tolstoy',
                      'CP08_18'],
                     ['Wuthering Heights by Emily Brontë', 'Sturmhöhe von Emily Brontë',
                      'CP08_19'],
                     ['Treasure Island by Robert L. Stevenson', 'Die Schatzinsel von Robert L. Stevenson',
                      'CP08_20']]

    book_id2english_title = {}
    for (english, german, index) in books_mapping:
        book_id2english_title[index.replace('CP08_', '')] = english

    languages = df["LANGUAGE"].tolist()

    # Numbers of Known Books
    numbers_of_known_books = df["CP08"].tolist()
    numbers_of_known_books_dict = defaultdict(list)
    for language, known_books in zip(languages, numbers_of_known_books):
        numbers_of_known_books_dict[language].append(known_books)
        numbers_of_known_books_dict["all"].append(known_books)

    language_books_dict = defaultdict(lambda: defaultdict(dict))
    print(numbers_of_known_books_dict)
    known_tuples = []
    for key, known_items in numbers_of_known_books_dict.items():
        if key == "eng":
            ttest = scipy.stats.ttest_ind(known_items, numbers_of_known_books_dict["ger"])
        elif key == "ger":
            ttest = scipy.stats.ttest_ind(known_items, numbers_of_known_books_dict["eng"])
        else:
            ttest = (None, None)
        known_tuples.append((key, np.mean(known_items), np.std(known_items), ttest[0], ttest[1]))

    book_knowledge_df = pd.DataFrame(known_tuples, columns=["Language", "Mean", "STD", "T-test stat", "T-test p"])
    book_knowledge_df.to_csv('results/human_assessment/book_knowledge.csv')
    # Overview to Book familarity with language sensitivity

    for column in df.columns:
        if str(column).startswith("CP08_"):
            prefix, suffix = column.split('_')
            answers = [books_answer_mapping[answer] for answer in df[column].tolist()]

            language_dict = defaultdict(list)
            for language, answer in zip(languages, answers):
                language_dict[language].append(answer)
            language_dict["all"].extend(language_dict["ger"])
            language_dict["all"].extend(language_dict["eng"])

            counter_language_dict = {}
            for language, language_books in language_dict.items():
                counter_dict = {"unknown": 0, "known": 0}
                for answer in language_books:
                    counter_dict[answer] += 1

                counter_language_dict[language] = counter_dict

            for key in language_dict:
                # language_books_dict[key][suffix].extend(language_dict[key])
                language_books_dict[key][suffix] = counter_language_dict[key]

    book_tuples = []
    for language, language_books in language_books_dict.items():
        for book_id, book_answers in language_books.items():
            book_tuples.append((book_id, book_id2english_title[book_id], language, book_answers["unknown"], book_answers["known"]))
    book_familarity_df = pd.DataFrame(book_tuples, columns=["Book ID", "Description", "Language", "Unknown", "Known"])
    book_familarity_df = book_familarity_df.sort_values(by=['Book ID']).set_index(["Book ID", "Description", "Language"])
    # print(book_familarity_df)
    book_familarity_df.to_csv('results/human_assessment/book_familarity.csv')

    # print('-----------------------------------------------------')
    facets = {
        "total": "GE",
        "location": "OR",
        "time": "ZE",
        "atmosphere": "AT",
        "plot": "PL",
        "content": "IN"
    }

    column_names = [str(column) for column in df.columns]
    triangulation_dict = defaultdict(lambda: defaultdict(list))
    triangulation_tuples = []
    for i, row in df.iterrows():
        # most_sim = -10.0
        book_tups = []
        for facet, facet_abb in facets.items():
            relevant_facet_columns = generate_facet_column_names(facet_abb)
            for selection_column, book_columns in relevant_facet_columns.items():
                if selection_column in df.columns:

                    books = []
                    for book_column in book_columns:
                        if book_column in df.columns:
                            books.append(str(row[book_column]).replace("CP08_", ""))
                    # print(selection_column, row[selection_column], books)
                    selection = selection_map(books[0],
                                              books[1],
                                              books[2],
                                              row[selection_column])
                    triangulation_dict[(books[0], books[1], books[2])][facet].append(selection)
                    if books[0] != "-10":
                        triangulation_tuples.append((books[0], books[1], books[2], facet, row["LANGUAGE"], row["CP09"],
                                                     selection))

    triangulation_df = pd.DataFrame(triangulation_tuples, columns=["Book 1", "Book 2", "Book 3", "Facet", "Language",
                                                                   "Group", "Selection"])
    triangulation_df.sort_values(['Book 1', 'Book 2', 'Book 3', 'Facet', 'Language', 'Group'], inplace=True)
    triangulation_df.to_csv('results/human_assessment/triangulation_raw.csv', index=False)

    print(triangulation_dict)

    kappa_dict = {}
    language_kappa = group_kappa_for_df(triangulation_df, "Language")
    group_kappa = group_kappa_for_df(triangulation_df, "Group")
    all_kappa = group_kappa_for_df(triangulation_df)
    facet_kappa = facet_kappa_for_df(triangulation_df)

    kappa_dict.update(language_kappa[0])
    kappa_dict.update(group_kappa[0])
    kappa_dict.update(all_kappa[0])
    kappa_dict.update(facet_kappa)
    print()
    print(kappa_dict)
    print()
    kappa_df = pd.DataFrame([(key, score) for key, score in kappa_dict.items()], columns=["Attribut", "Kappa Score"])
    kappa_df.to_csv('results/human_assessment/kappa_scores.csv', index=False)

    human_assessed_df = pd.DataFrame([(b1, b2, b3, facet, rating)
                                      for (b1, b2, b3, facet), rating in all_kappa[1].items()
                                      if rating != "skip" and rating != "unsure"],
                                     columns=["Book 1", "Book 2", "Book 3", "Facet", "Selection"])
    human_assessed_df.to_csv('results/human_assessment/human_assessed.csv', index=False)

        # for column_name, cell in zip(column_names, row):
        #     if column_name.startswith("GE") and "_" not in column_name:
        #         most_sim = int(cell)
        #         print(column_name, cell)
        #
        #     if column_name.startswith("GE") and "_" in column_name:
        #         book_tups.append((cell.replace("CP08_")))
        #     print(cell, book_tups)


    # language_answer_dict = defaultdict(lambda: defaultdict(list))
    # for column in df.columns:
    #     if str(column).startswith("CG"):
    #         prefix, suffix = column.split('_')
    #         answers = [comparison_answer_mapping[answer] for answer in df[column].tolist()]
    #
    #         language_dict = defaultdict(list)
    #         for language, answer in zip(languages, answers):
    #             if answer >= 0:
    #                 language_dict[language].append(answer)
    #         language_dict["all"].extend(language_dict["ger"])
    #         language_dict["all"].extend(language_dict["eng"])
    #         for key in language_dict:
    #             language_answer_dict[key][(prefix, comparison_suffix_mapping[suffix])].extend(language_dict[key])
    #         # answer_dict[(prefix, comparison_suffix_mapping[suffix])].extend(language_dict)
    #
    # # print(answer_dict["all"])
    # print(fleiss_kappa_self(kappa_matrix_of_answer_dict(language_answer_dict["all"])))
    # print(fleiss_kappa_self(kappa_matrix_of_answer_dict(language_answer_dict["ger"])))
    # # print(fleiss_kappa_self(kappa_matrix_of_answer_dict(answer_dict["eng"])))
    #
    # # aggregated values for each book pair and language
    # tuples = []
    # for language, answer_dict in language_answer_dict.items():
    #     # overal, german, english
    #     language_dict = defaultdict(list)
    #
    #     for comparison, answers in answer_dict.items():
    #         if len(answers) == 0:
    #             mean = None
    #             std = None
    #             fleiss_kappa_score = None
    #         else:
    #             answers = np.array(answers)
    #             mean = answers.mean()
    #             std = answers.std()
    #             fleiss_kappa_score = fleiss_kappa_self(answers_to_kappa_matrix(answers))
    #
    #         book_id_a = comp_id2book_id[comparison[0]][0]
    #         book_id_b = comp_id2book_id[comparison[0]][1]
    #         tuples.append((language, comparison[0], book_id_a, book_id2english_title[book_id_a], book_id_b,
    #                        book_id2english_title[book_id_b],
    #                        comparison[1], mean, std, fleiss_kappa_score))
    #
    # comparison_results_df = pd.DataFrame(tuples, columns=["Language", "Comparison ID",
    #                                                       "Book ID A", "Description Book A",
    #                                                       "Book ID B", "Description Book B",
    #                                                       "Similarity Type", "Mean",
    #                                                       "STD", "Fleiss Kappa"])
    #
    # # print(comparison_results_df)
    # comparison_results_df.to_csv('results/human_assessment/book_comparison.csv', index=False)




