from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from umap import UMAP

from corpus_structure import Corpus
from vectorization_utils import Vectorization
import random




def doc_id_replace(corpus: Corpus, doc_id: str):
    if doc_id[-1].isalpha():
        return f'{corpus.documents["_".join(doc_id.split("_")[:-1])].title} - {doc_id.split("_")[-1]}'
    else:
        return corpus.documents[doc_id].title


def colors(facet):
    if facet.endswith("loc"):
        return "#e69f00", "loc"
    elif facet.endswith("time"):
        return "#0072b2", "time"
    elif facet.endswith("sty"):
        return "#009e73", "sty"
    elif facet.endswith("raw"):
        return "#f0e442", "raw"
    elif facet.endswith("atm"):
        return "#cc79a7", "atm"
    elif facet.endswith("cont"):
        return "#d55e00", "cont"
    elif facet.endswith("plot"):
        return "#56b4e9", "plot"
    else:
        return "#000000", "sum"


def label_replace(facet):
    if facet == "loc":
        return "Location"
    elif facet == "time":
        return "Time"
    elif facet == "sty":
        return "Style"
    elif facet == "raw":
        return "Raw"
    elif facet == "atm":
        return "Atmosphere"
    elif facet == "cont":
        return "Content"
    elif facet == "plot":
        return "Plot"
    elif facet == "sum":
        return "Sum"
    else:
        return "Combine"


def tsne_plot(model, corpus: Corpus):
    labels = []
    tokens = []
    plt.rcParams.update({'font.size': 20})
    for doc_id in model.docvecs.doctags:
        if str(doc_id)[-1].isalpha():
            tokens.append(model.docvecs[doc_id])
            labels.append(doc_id_replace(corpus, doc_id))

    dim_reduced_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=500, random_state=42)
    # dim_reduced_model = UMAP(n_components=2, init='spectral', random_state=42)
    new_values = dim_reduced_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    dots = []
    labs = []
    for i in range(len(x)):
        if not labels[i].endswith("raw"):
            color, facet = colors(labels[i])
            ax = plt.scatter(x[i], y[i], c=color)
            if facet not in labs:
                dots.append(ax)
                labs.append(facet)
            # plt.annotate(labels[i].split(" - ")[0],
            #              xy=(x[i], y[i]),
            #              xytext=(5, 2),
            #              textcoords='offset points',
            #              ha='right',
            #              va='bottom')

    labs = [label_replace(lab) for lab in labs]
    plt.legend(dots,
               labs,
               # scatterpoints=1,
               loc='best',
               ncol=2,
               fontsize=20)

    plt.show()


def neighbor_plot(model, corpus: Corpus):
    labels = []
    vectors = []
    plt.rcParams.update({'font.size': 6})
    neighbors = {}
    for doc_id in model.docvecs.doctags:
        if str(doc_id)[-1].isalpha():
            sim_docs = Vectorization.most_similar_documents(model, corpus, positives=[doc_id], topn=2, print_results=False)
            # print(sim_docs)
            print(doc_id_replace(corpus, doc_id), doc_id_replace(corpus, sim_docs[-1][0]))

            sim_words = Vectorization.most_similar_words(model, corpus, positives=[doc_id], topn=2,
                                                         print_results=False)
            # print(doc_id, sim_docs)
            neighbors[doc_id] = sim_docs[-1][0]
            vectors.append(model.docvecs[doc_id])
            labels.append(doc_id_replace(corpus, doc_id))



    # dim_reduced_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=42)
    dim_reduced_model = UMAP(n_components=2, init='spectral', random_state=42)
    new_values = dim_reduced_model.fit_transform(vectors)

    reduced_dict = {label: new_value for new_value, label in zip(new_values, labels)}
    # print(reduced_dict)

    new_vals = []
    new_labels = []
    new_lines = []
    for doc_id in model.docvecs.doctags:
        if not doc_id.endswith("raw"):
            if "_0_" in doc_id or "_1_" in doc_id or "_2_" in doc_id or "_3_" in doc_id or True:
                try:
                    sim_doc_id = neighbors[doc_id]

                    lab = doc_id_replace(corpus, doc_id)
                    sim_lab = doc_id_replace(corpus, sim_doc_id)
                    dot = reduced_dict[lab]
                    sim_dot = reduced_dict[sim_lab]
                    new_vals.append(dot)
                    new_vals.append(sim_dot)
                    new_labels.append(lab)
                    new_labels.append(sim_lab)

                    x = [dot[0], sim_dot[0]]
                    y = [dot[1], sim_dot[1]]
                    c = colors(doc_id)[0]
                    new_lines.append((x, y, c))
                except KeyError:
                    pass

    new_values = new_vals
    labels = new_labels
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    dots = []
    labs = []
    for i in range(len(x)):
        color, facet = colors(labels[i])
        ax = plt.scatter(x[i], y[i], c=color)
        if facet not in labs:
            dots.append(ax)
            labs.append(facet)
        plt.annotate(labels[i].split(" - ")[0],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    for i, line in enumerate(new_lines):
        plt.plot(line[0], line[1], color=line[2])
    labs = [label_replace(lab) for lab in labs]

    plt.legend(dots,
               labs,
               # scatterpoints=1,
               loc='best',
               ncol=2,
               fontsize=20)

    plt.show()


if __name__ == '__main__':
    data_set_name = "classic_gutenberg"
    # data_set_name = "german_books"
    vectorization_algorithm = "book2vec_adv"
    vec_path = Vectorization.build_vec_file_name("all",
                                                 "no_limit",
                                                 data_set_name,
                                                 "no_filter",
                                                 vectorization_algorithm,
                                                 "real",
                                                 allow_combination=True)
    vectors, summation_method = Vectorization.my_load_doc2vec_format(vec_path)

    c = Corpus.fast_load("all",
                         "no_limit",
                         data_set_name,
                         "no_filter",
                         "real",
                         load_entities=False
                         )

    # tsne_plot(vectors, c)
    neighbor_plot(vectors, c)
