from collections import defaultdict
from typing import Dict, List, Set

from tqdm import tqdm

from corpus_structure import Corpus, CommonWords, Language

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
    c = Corpus(documents, name="Name", language=Language.DE)
    global_df = CommonWords.global_too_specific_words_doc_frequency(c, 1)
    print(global_df)
