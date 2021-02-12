import json
import os

from corpus_structure import Corpus, ConfigLoader

if __name__ == "__main__":
    config = ConfigLoader.get_config()
    corpus_to_annotate = "german_books"
    time_dict_path = os.path.join(config["system_storage"]["corpora"], 'plain_text', f'{corpus_to_annotate}_plain',
                                  'out', 'time_dict.json')
    with open(time_dict_path, encoding='utf-8') as json_file:
        data = json.load(json_file)
    x = Corpus.load_corpus_from_dir_format(os.path.join(config["system_storage"]["corpora"], corpus_to_annotate))
    x.update_time_entities(data)
    # x.save_corpus_adv(os.path.join(config["system_storage"]["corpora"], corpus_to_annotate))
