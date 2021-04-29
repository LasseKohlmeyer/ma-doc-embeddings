import logging
import os
import time

import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from KaggleWord2VecUtility import KaggleWord2VecUtility
from lib2vec.aux_utils import ConfigLoader
from lib2vec.corpus_iterators import CorpusSentenceIterator, CorpusPlainDocumentIterator
from lib2vec.corpus_structure import Corpus
from ksvd import ApproximateKSVD

config = ConfigLoader.get_config()


# import cPickle

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def dictionary_KSVD(num_clusters, word_vectors, basic_path: str):
    # Initalize a ksvd object and use it for clustering.
    aksvd = ApproximateKSVD(n_components=num_clusters)
    dictionary = aksvd.fit(word_vectors).components_
    idx_proba = aksvd.transform(word_vectors)
    idx = np.argmax(idx_proba, axis=1)
    # print("Clustering Done...", time.time() - start, "seconds")
    # Get probabilities of cluster assignments.

    # Dump cluster assignments and probability of cluster assignments.
    joblib.dump(idx, os.path.join(config["system_storage"]["models"], f'ksvd_{basic_path}'))
    print("Cluster Assignments Saved...")

    joblib.dump(idx_proba, os.path.join(config["system_storage"]["models"], f'ksvd_prob_{basic_path}'))
    print("Probabilities of Cluster Assignments Saved...")
    return (idx, idx_proba)


def dictionary_read_KSVD(idx_name, idx_proba_name):
    # Loads cluster assignments and probability of cluster assignments.
    idx = joblib.load(idx_name)
    idx_proba = joblib.load(idx_proba_name)
    print("Cluster Model Loaded...")
    return idx, idx_proba


def get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict, num_features, model,
                                 word_centroid_prob_map):
    # This function computes probability word-cluster vectors.

    prob_wordvecs = {}

    for word in word_centroid_map:
        prob_wordvecs[word] = np.zeros(num_clusters * num_features, dtype="float32")
        for index in range(0, num_clusters):
            try:
                prob_wordvecs[word][index * num_features:(index + 1) * num_features] = model.wv[word] * \
                                                                                       word_centroid_prob_map[word][
                                                                                           index] * word_idf_dict[word]
            except:
                continue
    return prob_wordvecs


def weight_building(weight_file, a_weight):
    f = open(weight_file, "rb")
    lines = f.readlines()
    weight_dict = {}
    total = 0
    for line in lines:
        word, count = line.split()[:2]
        weight_dict[word] = int(count)
        total = total + int(count)
    for word in weight_dict:
        prob = weight_dict[word] * 1.0 / total
        weight_dict[word] = a_weight * 1.0 / (a_weight * 1.0 + prob)
    return weight_dict


def create_weight_dict(model: Word2Vec, a_weight):
    total = 0
    weight_dict = {}
    for word in model.wv.vocab:
        count = model.wv.vocab[word].count
        weight_dict[word] = int(count)
        total = total + int(count)
    for word in weight_dict:
        prob = weight_dict[word] * 1.0 / total
        weight_dict[word] = a_weight * 1.0 / (a_weight * 1.0 + prob)
    return weight_dict


def create_cluster_vector_and_gwbowv(prob_wordvecs, weight_dict, wordlist, n_comp):
    # This function computes SDV feature vectors.
    bag_of_centroids = np.zeros(n_comp, dtype="float32")
    for word in wordlist:
        try:
            bag_of_centroids += prob_wordvecs[word] * weight_dict[word]
        except:
            pass
    norm = np.sqrt(np.einsum('...i,...i', bag_of_centroids, bag_of_centroids))
    if norm != 0:
        bag_of_centroids /= norm
    return bag_of_centroids


def pca_truncated_svd(X, X_test=None, n_comp=3):
    sklearn_pca = PCA(n_components=n_comp, svd_solver='full')
    X_pca = sklearn_pca.fit_transform(X)
    if X_test:
        X_pca_test = sklearn_pca.transform(X_test)
    else:
        X_pca_test = None
    del sklearn_pca
    return X_pca, X_pca_test


class PSIFVectors():
    num_features = 200  # int(sys.argv[1])  # Word vector dimensionality
    min_word_count = 20  # Minimum word count
    num_workers = 40  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    num_clusters = 40

    @classmethod
    def psif_w2v(cls, sentences: CorpusSentenceIterator, dat_set_name: str) -> str:

        model_name = f"{dat_set_name}_{cls.num_features}features_{cls.min_word_count}minwords_{cls.context}" \
                     f"context_len2alldata.w2v"
        model_name = os.path.join(config["system_storage"]["models"], model_name)
        print(model_name)
        if os.path.exists(model_name):
            print(model_name, "exists!")
            return model_name
        start = time.time()
        print(model_name, "does not exist!")
        # The csv file might contain very huge fields, therefore set the field_size_limit to maximum.
        # csv.field_size_limit(sys.maxsize)
        # Read train data.
        # train_word_vector = pd.read_pickle('all.pkl')
        # Use the NLTK tokenizer to split the paragraph into sentences.
        # tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        # sentences = []
        print("Parsing sentences from training set...")
        # Loop over each news article.
        # for review in train_word_vector["text"]:
        #     try:
        #         # Split a review into parsed sentences.
        #         sentences += KaggleWord2VecUtility.review_to_sentences(review, tokenizer)
        #     except:
        #         continue
        # sentences = CorpusSentenceIterator(corpus)
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        print("Training Word2Vec model...")
        # Train Word2Vec model.
        model = Word2Vec(sentences, workers=cls.num_workers, hs=0, sg=1, negative=10, iter=25,
                         size=cls.num_features, min_count=cls.min_word_count,
                         window=cls.context, sample=cls.downsampling, seed=1)
        model.init_sims(replace=True)
        # Save Word2Vec model.
        print("Saving Word2Vec model...")
        model.save(model_name)
        endmodeltime = time.time()
        print("time : ", endmodeltime - start)
        return model_name

    @classmethod
    def psif(cls, w2v_model_name: str, documents: CorpusPlainDocumentIterator, data_set_name: str):
        start = time.time()
        model_name = w2v_model_name
        # Load the trained Word2Vec model.
        model = Word2Vec.load(model_name)
        # Get wordvectors for all words in vocabulary.
        word_vectors = model.wv.vectors
        a_weight = 0.01
        # weight_file = "data/reuters_vocab.txt"
        # weight_dict = weight_building(weight_file, a_weight)
        weight_dict = create_weight_dict(model, a_weight)
        # Load all data.
        # all_data = pd.read_pickle('all.pkl')
        # Set number of clusters.

        # Uncomment below line for creating new clusters.
        # idx, idx_proba = dictionary_KSVD(num_clusters, word_vectors)
        basic_ksvd_path = f"latestclusmodel_len2alldata_{data_set_name}.pkl"
        for num_clusters in [40]:
            idx, idx_proba = dictionary_KSVD(num_clusters, word_vectors, basic_path=basic_ksvd_path)
            # for _ in range(10):
            #     data_all = pickle.load(open("data_all.pkl", "r"))
            #     all_x, Y = [], []
            #     for each in data_all:
            #         all_x.append(each[0])
            #         Y.append(each[1])
            #     train_data, test_data, Y_train, Y_test = train_test_split(all_x, Y, test_size=0.3,
            #                                                               random_state=random.randint(1, 100))

        # Uncomment below lines for loading saved cluster assignments and probabaility of cluster assignments.
        # idx_name = "ksvd_latestclusmodel_len2alldata.pkl"
        # idx_proba_name = "ksvd_prob_latestclusmodel_len2alldata.pkl"
        # idx, idx_proba = dictionary_read_KSVD(idx_name, idx_proba_name)

        # Create a Word / Index dictionary, mapping each vocabulary word to
        # a cluster number
        word_centroid_map = dict(zip(model.wv.index2word, idx))
        # Create a Word / Probability of cluster assignment dictionary, mapping each vocabulary word to
        # list of probabilities of cluster assignments.
        word_centroid_prob_map = dict(zip(model.wv.index2word, idx_proba))

        # building weighting dictionary for sif weighting

        # Computing tf-idf values.
        traindata = []
        for document in documents:
            traindata.append(" ".join(KaggleWord2VecUtility.review_to_wordlist(document, True)))

        tfv = TfidfVectorizer(strip_accents='unicode', dtype=np.float32)
        _ = tfv.fit_transform(traindata)
        featurenames = tfv.get_feature_names()
        idf = tfv._tfidf.idf_

        # Creating a dictionary with word mapped to its idf value
        print("Creating word-idf dictionary for Training set...")

        word_idf_dict = {}
        for pair in zip(featurenames, idf):
            word_idf_dict[pair[0]] = pair[1]

        # Pre-computing probability word-cluster vectors.
        prob_wordvecs = get_probability_word_vectors(featurenames, word_centroid_map, num_clusters, word_idf_dict,
                                                     cls.num_features, model,
                                                     word_centroid_prob_map)

        temp_time = time.time() - start
        print("Creating Document Vectors...:", temp_time, "seconds.")
        # Create train and text data.
        # lb = MultiLabelBinarizer()
        # Y = lb.fit_transform(all_data.tags)
        # train_data, test_data, Y_train, Y_test = train_test_split(all_data["text"], Y, test_size=0.3, random_state=42)

        # train = DataFrame({'text': []})
        # test = DataFrame({'text': []})
        #
        # train["text"] = train_data.reset_index(drop=True)
        # test["text"] = test_data.reset_index(drop=True)
        # gwbowv is a matrix which contains normalised document vectors.
        gwbowv = np.zeros((len(documents.corpus.documents), num_clusters * cls.num_features), dtype="float32")

        counter = 0
        n_comp = cls.num_features * num_clusters
        for document in documents:

            # Get the wordlist in each news article.
            words = KaggleWord2VecUtility.review_to_wordlist(document, remove_stopwords=True)
            gwbowv[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, weight_dict, words, n_comp)
            counter += 1
            if counter % 1000 == 0:
                print("Train text Covered : ", counter)

        doc_ids = documents.doc_ids

        gwbowv_name = "SDV_" + str(num_clusters) + "cluster_" + str(cls.num_features) + "feature_matrix_ksvd_sparse.npy"

        # gwbowv_test = np.zeros((test["text"].size, num_clusters * num_features), dtype="float32")

        # counter = 0

        # for review in test["text"]:
        #     # Get the wordlist in each news article.
        #     words = KaggleWord2VecUtility.review_to_wordlist(review,
        #                                                      remove_stopwords=True)
        #     gwbowv_test[counter] = create_cluster_vector_and_gwbowv(prob_wordvecs, weight_dict, words, n_comp)
        #     counter += 1
        #     if counter % 1000 == 0:
        #         print("Test text Covered : ", counter)

        # test_gwbowv_name = "TEST_SDV_" + str(num_clusters) + "cluster_" + str(
        #     num_features) + "feature_matrix_ksvd_sparse.npy"

        words_dict = None
        if len(doc_ids) >= cls.num_features:
            gwbowv, gwbowv_test = pca_truncated_svd(gwbowv, None, cls.num_features)
            print(gwbowv.shape)
            # saving gwbowv train and test matrices
            np.save(gwbowv_name, gwbowv)
            words_dict = {word: model.wv[word] for word in model.wv.vocab}

        docs_dict = {doc_id: vec for doc_id, vec in zip(doc_ids, gwbowv)}

        # np.save(test_gwbowv_name, gwbowv_test)

        endtime = time.time() - start

        print("SDV created and dumped: ", endtime, "seconds.")

        print("********************************************************")
        return docs_dict, words_dict

    @classmethod
    def calculate_psif(cls, sentences, documents, data_set_name):
        model_path = cls.psif_w2v(sentences, data_set_name)
        return cls.psif(model_path, documents, data_set_name)


if __name__ == '__main__':
    data_set = "classic_gutenberg"
    # data_set_name = "german_books"
    # filter = "specific_words_strict"  # "no_filter"
    c = Corpus.fast_load("all",
                         "no_limit",
                         data_set,
                         "no_filter",
                         "real",
                         load_entities=False
                         )
    sents = CorpusSentenceIterator(c)
    docs = CorpusPlainDocumentIterator(c)

    dd, wd = PSIFVectors.calculate_psif(sents, docs, data_set)
    for d, v in dd.items():
        print(d, v.shape)
