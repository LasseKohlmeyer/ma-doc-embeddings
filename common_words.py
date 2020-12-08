from collections import defaultdict
from typing import Dict, List, Set

from tqdm import tqdm


class CommonWords:
    @staticmethod
    def testing(dict_a: Dict[str, Set[str]], dict_b: Dict[str, Set[str]]):
        assert dict_a.keys() == dict_b.keys()
        for series_id in dict_a:
            set_a = dict_a[series_id]
            set_b = dict_b[series_id]
            assert set_a == set_b

    @staticmethod
    def without_gerneral_words(common_words: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        medium_common_words = {}
        for series_id_a, series_words_a in common_words.items():
            series_words_a_copy = set()
            series_words_a_copy.update(series_words_a)
            general_words = set()
            for series_id_b, series_words_b in common_words.items():
                if series_id_a != series_id_b:
                    general_words.update(series_words_a_copy.intersection(series_words_b))

            series_words_a_copy.difference_update(general_words)
            medium_common_words[series_id_a] = series_words_a_copy
        return medium_common_words

    @staticmethod
    def strict(series_dictionary: Dict[str, List[str]], doc_texts: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = defaultdict(set)
        for series_id, doc_ids in series_dictionary.items():
            series_words = []
            for doc_id in doc_ids:
                series_words.append(set(doc_texts[doc_id]))

            for token_set_a in series_words:
                for token_set_b in series_words:
                    if token_set_a != token_set_b:
                        common_words[series_id].update(token_set_a.intersection(token_set_b))
        return dict(common_words)

    @staticmethod
    def strict_general_words_sensitive(series_dictionary: Dict[str, List[str]], doc_texts: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = CommonWords.strict(series_dictionary, doc_texts)
        medium_common_words = CommonWords.without_gerneral_words(common_words)
        return medium_common_words

    @staticmethod
    def relaxed(series_dictionary: Dict[str, List[str]], doc_texts: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        common_words = defaultdict(set)
        for series_id, doc_ids in series_dictionary.items():
            series_words = []
            for doc_id in doc_ids:
                series_words.append(set(doc_texts[doc_id]))
            common_words[series_id] = set.intersection(*series_words)

        return dict(common_words)

    @staticmethod
    def relaxed_general_words_sensitive(series_dictionary: Dict[str, List[str]], doc_texts: Dict[str, List[str]]) \
            -> Dict[str, Set[str]]:
        common_words = CommonWords.relaxed(series_dictionary, doc_texts)
        medium_common_words = CommonWords.without_gerneral_words(common_words)
        return medium_common_words

    @staticmethod
    def global_common_words(doc_texts: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        tokens = [set(doc_tokens) for doc_id, doc_tokens in doc_texts.items()]

        global_intersect = set()
        for token_set_a in tokens:
            for token_set_b in tokens:
                if token_set_a != token_set_b:
                    global_intersect.update(token_set_a.intersection(token_set_b))

        global__strict_intersect = set.intersection(*tokens)

        common_words = {}
        for c, (doc_id, doc_tokens) in enumerate(doc_texts.items()):
            not_to_delete = set(doc_tokens).difference(global_intersect).union(global__strict_intersect)
            to_delete = set(doc_tokens).difference(not_to_delete)
            common_words[doc_id] = to_delete

        return common_words

    @staticmethod
    def global_too_specific_words_doc_frequency(doc_texts: Dict[str, List[str]], percentage_share: float,
                                                absolute_share: int = None) \
            -> Dict[str, Set[str]]:
        # d = defaultdict(set)
        # for doc_id, doc_tokens in doc_texts.items():
        #     for token in doc_tokens:
        #         d[token].add(doc_id)
        # freq_dict = {token: len(doc_ids) / len(doc_texts) for token, doc_ids in d.items()}
        tqdm_disable = True
        freq_dict = defaultdict(lambda: 0.0)
        vocab = set()
        if absolute_share:
            percentage_share = absolute_share
        for doc_id, doc_tokens in tqdm(doc_texts.items(), total=len(doc_texts), desc="Calculate DF",
                                       disable=tqdm_disable):

            for token in set(doc_tokens):
                vocab.add(token)
                if absolute_share:
                    freq_dict[token] += 1
                else:
                    freq_dict[token] += 1 / len(doc_texts)
                    if freq_dict[token] > 1:
                        freq_dict[token] = 1

                    # if not lower_bound:
        #     lower_bound = lower_bound_absolute / len(doc_texts)
        # print(len(freq_dict.keys()), len(vocab))
        # print(min(freq_dict.values()), max(freq_dict.values()))
        to_remove = [token for token, doc_freq in tqdm(freq_dict.items(), total=len(freq_dict), desc="Filter DF",
                                                       disable=tqdm_disable)
                     if doc_freq <= percentage_share]
        # print(len(to_remove))
        # print(to_remove)
        too_specific_words = {doc_id: set(doc_tokens).intersection(to_remove)
                              for doc_id, doc_tokens in tqdm(doc_texts.items(), total=len(doc_texts),
                                                             desc="Extract words to remove", disable=tqdm_disable)}

        new_toks = set()
        for doc_id, toks in too_specific_words.items():
            new_toks.update(toks)
        # print(len(new_toks))
        # opposite
        # too_specific_words = {doc_id: set(doc_tokens).difference(to_remove)
        #                 for doc_id, doc_tokens in doc_texts.items()}
        # print('>', too_specific_words)
        return too_specific_words


if __name__ == "__main__":
    documents = {'d_0_0': 'A B C D F G H', 'd_0_1': 'A B C D G I W 1', 'd_0_2': 'A B C D F J W 1 2',
                 'd_1_0': 'A X Y Z V 2', 'd_1_1': 'A B C X V 1', 'd_1_2': 'A G Z V W 1 2',
                 'd_2_0': '1 2 3 4 5 A', 'd_2_1': 'A B 1 2 3 4', 'd_2_2': 'A G Z V Y 0'}
    series_dict = {'d_0': ['d_0_0', 'd_0_1', 'd_0_2'],
                   'd_1': ['d_1_0', 'd_1_1', 'd_1_2'],
                   'd_2': ['d_2_0', 'd_2_1', 'd_2_2']}

    str_res = {'d_0': {'A', 'B', 'C', 'D', 'F', 'G', 'W', '1'},
               'd_1': {'A', 'X', 'Z', 'V', '1', '2'},
               'd_2': {'A', '2', '4', '3', '1'}
               }  # correct

    strict_med_res = {'d_0': {'B', 'C', 'D', 'F', 'G', 'W'},
                      'd_1': {'X', 'Z', 'V'},
                      'd_2': {'3', '4'}
                      }  # correct

    rel_med_res = {'d_0': {'A', 'B', 'C', 'D'},
                   'd_1': {'A', 'V'},
                   'd_2': {'A'}
                   }  # correct

    rel_res = {'d_0': {'B', 'C', 'D'},
               'd_1': {'V'},
               'd_2': set()
               }  # correct

    global_res = {'d_0_0': {'C', 'G', 'F', 'D', 'B'}, 'd_0_1': {'C', 'G', '1', 'W', 'D', 'B'}, 'd_0_2': {'C', '2', 'F', '1', 'W', 'D', 'B'},
                  'd_1_0': {'X', '2', 'V', 'Z', 'Y'}, 'd_1_1': {'C', 'X', '1', 'V', 'B'}, 'd_1_2': {'G', '2', '1', 'W', 'V', 'Z'},
                  'd_2_0': {'3', '2', '4', '1'}, 'd_2_1': {'3', '2', '1', 'B', '4'}, 'd_2_2': {'G', 'V', 'Z', 'Y'}}

    documents = {doc_id: document.split() for doc_id, document in documents.items()}
    # str_cw = CommonWords.strict(series_dict, documents)
    # str_med_cw = CommonWords.strict_general_words_sensitive(series_dict, documents)
    # rel_med_cw = CommonWords.relaxed(series_dict, documents)
    # rel_cw = CommonWords.relaxed_general_words_sensitive(series_dict, documents)

    # print(str_cw)
    # print(str_med_cw)
    # print(rel_med_cw)
    # print(rel_cw)
    #
    # CommonWords.testing(str_res, str_cw)
    # CommonWords.testing(strict_med_res, str_med_cw)
    # CommonWords.testing(rel_med_res, rel_med_cw)
    # CommonWords.testing(rel_res, rel_cw)
    # global_cw = CommonWords.global_common_words(documents)
    # print(global_cw)
    # CommonWords.testing(global_res, global_cw)
    global_df = CommonWords.global_too_specific_words_doc_frequency(documents, 1)
    print(global_df)
