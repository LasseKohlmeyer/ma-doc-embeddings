import json
import os
import HeidelTime
from bs4 import BeautifulSoup

from utils import Language, ConfigLoader


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


def tag_file(input_file_name: str):
    lan = "english"
    if Language.get_from_str(input_file_name.replace('.txt', '').split('_')[-1]) == Language.DE:
        lan = "german"
    hw = HeidelTime.HeidelTimeWrapper(lan, doc='narratives')
    tagged_tokens = set()
    with open(input_file_name, 'r', encoding="utf-8") as file:
        lines = file.readlines()
        chunked_lines = chunks(lines, 100)
        for chunk in chunked_lines:
            string_to_parse = '\n'.join(chunk)
            # print(string_to_parse)
            soup = BeautifulSoup(str(hw.parse(string_to_parse)), "html.parser")
            tagged_tokens.update([i.text for i in soup.find_all('timex3')])
            # print(tagged_tokens)
    return list(tagged_tokens)


if __name__ == "__main__":
    config = ConfigLoader.get_config()
    corpus_to_annotate = "german_series"
    # input_dir = os.path.join(corpora/plain_text/german_series_plain)
    input_dir = os.path.join(config["system_storage"]["corpora"], 'plain_text', f'{corpus_to_annotate}_plain')
    path_names = [text_file for text_file in os.listdir(input_dir) if text_file.endswith('.txt')]
    new_dir = os.path.join(input_dir, "out")
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    time_dict = {}
    for path_name in path_names:
        if str(path_name).endswith('.txt'):
            input_file_path = os.path.join(input_dir, path_name)
            time_dict['_'.join(path_name.split('_')[:-1])] = tag_file(input_file_path)
            with open(os.path.join(new_dir, 'time_dict.json'), 'w', encoding="utf-8") as json_file:
                json.dump(time_dict, json_file, indent=1, ensure_ascii=False)

    print(time_dict)

    with open(os.path.join(new_dir, 'time_dict.json'), 'w', encoding="utf-8") as json_file:
        json.dump(time_dict, json_file, indent=1, ensure_ascii=False)
