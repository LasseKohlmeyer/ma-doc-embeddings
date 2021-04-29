import json
import math
import os
import HeidelTime
from bs4 import BeautifulSoup
from tqdm import tqdm

from lib2vec.corpus_structure import Language, ConfigLoader


def get_time_tokens(input_string: str, language: Language):
    lan = "english"
    if language == Language.DE:
        lan = "german"
    hw = HeidelTime.HeidelTimeWrapper(lan, doc='narratives')
    soup = BeautifulSoup(str(hw.parse(input_string)), "html.parser")
    return [i.text for i in soup.find_all('timex3')]


# def time_dict_for_corpus(corpus: Corpus):
#     time_dict = defaultdict(set)
#     for doc_id, document in tqdm(corpus.documents.items(), total=len(corpus.documents),
#                                  desc="Heideltime documents"):
#
#         for sentence in document.sentences:
#             time_dict[doc_id].update(get_time_tokens(sentence.representation(), document.language))

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# def tag_file(input_file_name: str):
#     lan = "english"
#     if Language.get_from_str(input_file_name.replace('.txt', '').split('_')[-1]) == Language.DE:
#         lan = "german"
#     hw = HeidelTime.HeidelTimeWrapper(lan, doc='narratives')
#     tagged_tokens = set()
#     with open(input_file_name, 'r', encoding="utf-8") as file:
#         print(input_file_name)
#         lines = file.readlines()
#         chunked_lines = chunks(lines, 100)
#         for chunk in chunked_lines:
#             string_to_parse = '\n'.join(chunk)
#             # print(string_to_parse)
#             soup = BeautifulSoup(str(hw.parse(string_to_parse)), "html.parser")
#             tagged_tokens.update([i.text for i in soup.find_all('timex3')])
#             # print(tagged_tokens)
#     return list(tagged_tokens)

def beboop(chunk):
    string_to_parse = ''.join(chunk)
    # print(indicator, "/", maximal, input_file_name, i + 1, "/", math.ceil(len(lines) / chunk_size),
    #       len(string_to_parse))
    # # print()
    # print(string_to_parse)
    soup = BeautifulSoup(str(hw.parse(string_to_parse)), "html.parser")
    return [i.text for i in soup.find_all('timex3')]


def tag_file(input_file_name: str, hw: HeidelTime.HeidelTimeWrapper, indicator: int, maximal: int):
    tagged_tokens = set()
    with open(input_file_name, 'r', encoding="utf-8") as file:
        # print(input_file_name)
        lines = file.readlines()
        chunk_size = 50
        chunked_lines = chunks(lines, chunk_size)
        for i, chunk in enumerate(chunked_lines):
            string_to_parse = ''.join(chunk)
            print(indicator, "/", maximal, input_file_name, i+1, "/", math.ceil(len(lines) / chunk_size), len(string_to_parse))
            # print()
            # print(string_to_parse)
            soup = BeautifulSoup(str(hw.parse(string_to_parse)), "html.parser")
            tagged_tokens.update([i.text for i in soup.find_all('timex3')])
            # print(tagged_tokens)

        # sets = Parallel(n_jobs=8)(delayed(beboop)(chunk)
        #                           for i, chunk in enumerate(chunked_lines))
    return list(tagged_tokens)


if __name__ == "__main__":
    config = ConfigLoader.get_config()

    corpus_to_annotate = "dta"
    # lan = "english"
    lan = "german"

    # input_dir = os.path.join(corpora/plain_text/german_series_plain)
    input_dir = os.path.join(config["system_storage"]["corpora"], 'plain_text', f'{corpus_to_annotate}_plain')
    path_names = [text_file for text_file in os.listdir(input_dir) if text_file.endswith('.txt')]
    new_dir = os.path.join(input_dir, "out")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    time_dict = {}

    hw = HeidelTime.HeidelTimeWrapper(lan, doc='narratives')
    c = 0
    time_dict_path = os.path.join(new_dir, 'time_dict.json')
    if not os.path.isfile(time_dict_path):
        already_processed = set()
    else:
        with open(os.path.join(new_dir, 'time_dict.json'), encoding="utf-8") as json_file:
            old_data = json.load(json_file)
            already_processed = old_data.keys()
        time_dict.update(old_data)
    for path_name in tqdm(path_names, total=len(path_names), desc="Files completed", disable=True):
        doc_id = '_'.join(path_name.split('_')[:-1])
        print(doc_id)
        if str(path_name).endswith('.txt'):
            if doc_id in already_processed:
                c += 1
                continue
            else:
                # print(doc_id, already_processed)
                input_file_path = os.path.join(input_dir, path_name)
                time_dict[doc_id] = tag_file(input_file_path, hw, c, len(path_names))
                with open(os.path.join(new_dir, 'time_dict.json'), 'w', encoding="utf-8") as json_file:
                    json.dump(time_dict, json_file, indent=1, ensure_ascii=False)
                c += 1

    # print(time_dict)

    with open(os.path.join(new_dir, 'time_dict.json'), 'w', encoding="utf-8") as json_file:
        json.dump(time_dict, json_file, indent=1, ensure_ascii=False)
