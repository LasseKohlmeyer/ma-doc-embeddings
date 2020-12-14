import math
import os

from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import json
from corpus_structure import Document, Corpus, Language


class Summarizer:
    @staticmethod
    def read_article(file_name):
        file = open(file_name, "r", encoding="utf-8")
        filedata = ' '.join(file.readlines())
        # print(filedata)
        article = filedata.split(". ")
        sentences = []

        for sentence in article:
            # print(sentence)
            sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
        sentences.pop()

        return sentences

    @staticmethod
    def sentence_similarity(sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    @staticmethod
    def build_similarity_matrix(sentences, stop_words):
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:  # ignore if both are same sentences
                    continue
                similarity_matrix[idx1][idx2] = Summarizer.sentence_similarity(sentences[idx1],
                                                                               sentences[idx2],
                                                                               stop_words)

        return similarity_matrix

    @staticmethod
    def generate_summary(file_name, top_n=5):
        stop_words = stopwords.words('german')
        summarize_text = []

        # Step 1 - Read text anc split it
        sentences = Summarizer.read_article(file_name)

        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = Summarizer.build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        print("Indexes of top ranked_sentence order are ", ranked_sentence)

        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize texr
        print("Summarize Text: \n", ". ".join(summarize_text))

    @staticmethod
    def summarize_sentences_rec(sentences, stop_words, top_n):
        # def chunks(lst, n):
        #     """Yield successive n-sized chunks from lst."""
        #     for i in range(0, len(lst), n):
        #         yield lst[i:i + n]
        def chunks(a, n):
            k, m = divmod(len(a), n)
            return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

        def chunk_nr(nr_sentences: int):
            val = math.floor(math.log(nr_sentences, 10))
            if val < 2:
                val = 2
            return val

        if len(sentences) <= 100:
            sentence_similarity_martix = Summarizer.build_similarity_matrix(sentences, stop_words)
            sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
            scores = nx.pagerank_numpy(sentence_similarity_graph)
            ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
            candidates = [ranked_sent[1] for ranked_sent in ranked_sentence]

            return candidates[:top_n]
        else:
            chunked_sentences = chunks(sentences, chunk_nr(len(sentences)))
            summarized_sentences = []
            for chunk in chunked_sentences:
                summarized_chunk_sents = Summarizer.summarize_sentences_rec(chunk, stop_words, top_n)
                # print('>>>', len(summarized_chunk_sents))
                summarized_sentences.extend(summarized_chunk_sents)
            # print('>>', len(summarized_sentences))
            summarized_sentences = Summarizer.summarize_sentences_rec(summarized_sentences, stop_words, top_n)
            # print('>', len(summarized_sentences))
            # print(summarized_sentences)
            return summarized_sentences[:top_n]

    @staticmethod
    def summarize_sentences_lin(sentences, stop_words, top_n):
        print(len(sentences))
        # Step 2 - Generate Similary Martix across sentences
        sentence_similarity_martix = Summarizer.build_similarity_matrix(sentences, stop_words)

        # Step 3 - Rank sentences in similarity martix
        sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
        scores = nx.pagerank(sentence_similarity_graph)

        # Step 4 - Sort the rank and pick top sentences
        ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        print("Indexes of top ranked_sentence order are ", ranked_sentence)

        summarize_text = []
        for i in range(top_n):
            summarize_text.append(" ".join(ranked_sentence[i][1]))

        # Step 5 - Offcourse, output the summarize texr
        print("Summarize Text: \n", ". ".join(summarize_text))
        return [ranked_sentence[i][1] for i in range(top_n)]

    @staticmethod
    def generate_summary_of_corpus_doc(document: Document, top_n=5, as_sent_ids: bool = True):
        stop_words = set(stopwords.words('english'))
        if document.language == Language.DE:
            stop_words = set(stopwords.words('german'))
        # doc_stop_words = [token.representation() for sentence in document.sentences
        #                   for token in sentence.tokens if token.stop]
        # stop_words.update(doc_stop_words)
        stop_words = list(stop_words)

        # Step 1 - Read text anc split it
        orig_sentences = [[token.representation() for token in sentence.tokens] for sentence in document.sentences]
        sentences = [[token.representation() for token in sentence.tokens] for sentence in document.sentences
                     if len(sentence.tokens) > 10]
        # print(len(stop_words), len(sentences))
        # sentences = read_article('msft.txt')
        res = Summarizer.summarize_sentences_rec(sentences, stop_words, top_n=top_n)
        # print(res)
        # print('-->', len(res))
        sent_ids = []
        if as_sent_ids:
            for res_sentence in res:
                for sent_nr, sentence in enumerate(orig_sentences):
                    if sentence == res_sentence:
                        sent_ids.append(sent_nr)
                        # print(sent_nr, document.sentences[sent_nr].representation(), res_sentence)
                        assert (document.sentences[sent_nr].representation() == res_sentence)
        # print('-->>', len(sent_ids))
        # for r in res:
        #     print('>>',  ' '.join(r))
        # res = summarize_sentences_lin(sentences, stop_words, top_n=top_n)
        # for r in res:
        #     print('>>',  ' '.join(r))

        # print(len(res), sent_ids, res)
        for sent_id, o_sent in zip(sent_ids, res):
            if document.sentences[sent_id].representation() != o_sent:
                print("Not correct", document.sentences[sent_id].representation(), o_sent)
        return res, sent_ids

    @staticmethod
    def get_summary(corpus_root_path: str):
        summary_dict_path = os.path.join(corpus_root_path, "sent_ids.json")
        if not os.path.isfile(summary_dict_path):
            summary_dict = {}
            root_corpus = Corpus.fast_load(path=corpus_root_path)
            for doc_id, doc in root_corpus.documents.items():
                sents, ids = Summarizer.generate_summary_of_corpus_doc(doc, 20)
                # print(doc_id, ":", ids, [' '.join(sent) for sent in sents])
                summary_dict[doc_id] = ids
            with open(summary_dict_path, 'w', encoding='utf-8') as fp:
                json.dump(summary_dict, fp, indent=1)
        else:
            with open(summary_dict_path) as json_file:
                summary_dict = json.load(json_file)
        return summary_dict

    @staticmethod
    def get_corpus_summary_sentence_list(corpus: Corpus, lemma: bool, lower: bool):
        corpus_summary = []
        if corpus.root_corpus_path is None:
            raise UserWarning("No root corpus set!")
        summary_corpus_path = corpus.root_corpus_path
        corpus_summary_dict = Summarizer.get_summary(summary_corpus_path)
        _, doc_ids = corpus.get_texts_and_doc_ids()
        for doc_id in doc_ids:
            doc_summary_tokens = []
            for sentence_id in corpus_summary_dict[doc_id]:
                # print(corpus.documents[doc_id].sentences[sentence_id].representation())
                doc_summary_tokens.extend([token.representation(lemma=lemma, lower=lower)
                                           for token in corpus.documents[doc_id].sentences[sentence_id].tokens])
            corpus_summary.append(doc_summary_tokens)
        return corpus_summary


if __name__ == "__main__":
    pass
    # c = Corpus.load_corpus_from_dir_format(os.path.join("corpora/german_series_all_no_limit_stopwords_real_"))
    # # summary_d = Summarizer.get_summary(c, os.path.join("corpora/german_series"))
    # # print(summary_d)
    # print(c.root_corpus_path)
    # print(Summarizer.get_corpus_summary_sentence_list(c, False, False))
    # print(c.name)
