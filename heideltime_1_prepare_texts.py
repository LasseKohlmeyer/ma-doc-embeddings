import os

from utils import ConfigLoader, Corpus


def corpus2plain_text_dir(source_path: str):
    corpus = Corpus.fast_load(path=source_path)

    new_dir = os.path.join(config["system_storage"]["corpora"], 'plain_text', f'{os.path.basename(source_path)}_plain')
    print(new_dir)
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    for doc_id, d in corpus.documents.items():
        doc_path = os.path.join(new_dir, f'{doc_id}_{d.language}.txt')
        with open(doc_path, 'w', encoding="utf-8") as writer:
            writer.write('\n'.join([' '.join(sent.representation()) for sent in d.sentences]))


if __name__ == "__main__":
    corpus_to_annotate = "german_series"
    config = ConfigLoader.get_config()
    corpus2plain_text_dir(os.path.join(config["system_storage"]["corpora"], corpus_to_annotate))