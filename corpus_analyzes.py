from collections import defaultdict, Counter
from series_prove_of_concept import EvaluationMath
import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def rreplace(s, old, new, occurrence):
    li = s.rsplit(old, occurrence)
    return new.join(li)


def group_color_value(nr_groups, group_nr):
    if nr_groups == 1:
        return 1.0
    else:
        return 1.0 / (nr_groups-1) * group_nr


def group_distribution(input_dict, special_colors: bool = True):
    # cmap = plt.cm.get_cmap('brg')
    cmap = plt.cm.get_cmap('viridis')

    for outer_group_name, outer_values in input_dict.items():
        nr_inner_groups = len(outer_values.keys())
        for i, (inner_group_name, inner_values) in enumerate(outer_values.items()):
            rgba = cmap(group_color_value(nr_groups=nr_inner_groups, group_nr=i))
            c = Counter(inner_values)
            print(outer_group_name, inner_group_name, c)
            if special_colors:
                plt.hist(c, alpha=0.4, bins=100, label=inner_group_name, color=rgba)
            else:
                plt.hist(c, alpha=0.4, bins=100, label=inner_group_name)

        plt.legend(loc='upper right')
        plt.title(outer_group_name)
        plt.show()


def group_medians_plot(df: pd.DataFrame, group: str, pivot_col: str, median: bool = True):
    df_grouped = df.groupby([group, pivot_col])
    stacked = False
    # if group == "segment_id":
    #     stacked = True
    # else:
    #     stacked = False
    if median:
        avg_fun = "Median"
        avg_error = "IQR"
        agg_df = df_grouped.median()
        agg_df.columns = [avg_fun]
        agg_df[[avg_error]] = df_grouped.quantile(0.75) - df_grouped.quantile(0.25)
    else:
        avg_fun = "Mean"
        avg_error = "STD"
        agg_df = df_grouped.mean()
        agg_df.columns = [avg_fun]
        agg_df[avg_error] = df_grouped.std()
    # agg_df = agg_df.reset_index()
    # agg_df.set_index(agg_df[group])
    print(agg_df.head())
    agg_df = agg_df.unstack()
    print(agg_df)
    ax = agg_df[avg_fun].plot(kind='bar', stacked=stacked, yerr=agg_df[avg_error])
    ax.set_ylim(0)
    # agg_df["Median"].plot.bar(yerr=agg_df["IQR"])
    plt.show()


def parallel_coords(df: pd.DataFrame, group: str, base_index: str = None):
    if base_index is not None:
        df = df.drop([base_index], axis=1)
    pd.plotting.parallel_coordinates(df, class_column=group, colormap="Dark2")
    plt.show()


def pivot(df: pd.DataFrame, pivot_col: str, group: str, value_col: str, base_index: str = None):
    if base_index is not None:
        index_cols = [base_index, group]
    else:
        index_cols = [group]
    df = df.set_index(index_cols)
    # df['a'] = pd.Categorical(df[pivot_col], categories=df[pivot_col].unique(), ordered=True)
    df = df.pivot(columns=pivot_col)[value_col]
    df.reset_index(inplace=True)

    return df


def aggregating(df: pd.DataFrame, pivot_col: str, group: str, value_col: str, meta_col: str = None):
    if meta_col:
        agg_df = df.groupby([meta_col, group, pivot_col]).median()
    else:
        agg_df = df.groupby([group, pivot_col]).median()

    agg_df = agg_df.reset_index()

    if meta_col:
        # agg_df = pivot(agg_df, pivot_col=pivot_col, base_index="file_name", group=group, value_col=value_col)
        agg_df = agg_df.set_index([meta_col, group])
        # df['a'] = pd.Categorical(df[pivot_col], categories=df[pivot_col].unique(), ordered=True)
        agg_df = agg_df.pivot(columns=pivot_col)[value_col]
        agg_df.reset_index(inplace=True)
        # df.fillna(0)
    else:
        agg_df = pivot(agg_df, pivot_col=pivot_col, base_index=None, group=group, value_col=value_col)

    return agg_df


def analyze_word_distribution(all_dict, absoulte: bool = True):
    all_tuples = []
    for name, file_dict in all_dict.items():

        segment_dict = defaultdict(lambda: defaultdict(list))
        aspect_dict = defaultdict(lambda: defaultdict(list))
        corpus_wise_doc_tuples = []
        for doc_id, doc_aspects in file_dict.items():
            segment_id = doc_id.split('_')[-1]
            base_doc_id = rreplace(doc_id, f'_{segment_id}', '', 1)
            doc_len = sum([value for value in doc_aspects.values()])
            # print(doc_id, doc_len)
            for aspect, value in doc_aspects.items():
                if absoulte:
                    val = value
                else:
                    val = value/doc_len

                segment_dict[segment_id][aspect].append(val)
                aspect_dict[aspect][segment_id].append(val)

                corpus_wise_doc_tuples.append((base_doc_id, segment_id, val, aspect))
                all_tuples.append((name, base_doc_id, segment_id, val, aspect))
        doc_df = pd.DataFrame.from_records(corpus_wise_doc_tuples, columns=["doc_id", "segment_id", "value", "aspect"])

        # print dicts
        for segment_id, aspects in segment_dict.items():
            for aspect, values in aspects.items():
                print(segment_id, aspect, EvaluationMath.median(values), EvaluationMath.mean(values))

        for aspect, segments in aspect_dict.items():
            for segment_id, values in segments.items():
                print(aspect, segment_id, EvaluationMath.median(values), EvaluationMath.mean(values))
        group_medians_plot(doc_df, group="aspect", pivot_col="segment_id")
        group_medians_plot(doc_df, group="segment_id", pivot_col="aspect")

        # agg_segments = aggregating(doc_df, pivot_col="segment_id", group="aspect", value_col="value")
        # parallel_coords(agg_segments, group="aspect", base_index=None)

        segment_df = pivot(doc_df, pivot_col='segment_id', base_index='doc_id', group='aspect', value_col='value')
        parallel_coords(segment_df, group="aspect", base_index='doc_id')

        # agg_aspects = aggregating(doc_df, pivot_col="aspect", group="segment_id",  value_col="value")
        # parallel_coords(agg_aspects, group="segment_id", base_index=None)

        aspect_df = pivot(doc_df, pivot_col='aspect', base_index='doc_id', group='segment_id', value_col='value')
        parallel_coords(aspect_df,  group="segment_id", base_index='doc_id')

        # group_distribution(aspect_dict)
        # group_distribution(segment_dict)

    # all_df = pd.DataFrame.from_records(all_tuples, columns=["file_name", "doc_id", "segment_id", "value", "aspect"])
    # print(all_df.head())
    # agg_segments = aggregating(all_df, pivot_col="segment_id", group="aspect", value_col="value",
    #                            meta_col="file_name")
    # parallel_coords(agg_segments, group="aspect", base_index="file_name")

    # agg_aspects = aggregating(all_df, pivot_col="aspect", group="segment_id", value_col="value", meta_col="file_name")
    # parallel_coords(agg_aspects, group="segment_id", base_index="file_name")
    #
    # segment_df = pivot(all_df, pivot_col='segment_id', base_index='doc_id', group='aspect', value_col='value')
    # parallel_coords(segment_df, group="aspect", base_index='doc_id')
        # (,*[segments] aspect)


if __name__ == '__main__':
    path_to_json = 'aspects/'
    corpus_select = 'german_series'
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if '_no_filter_real_book2vec_adv.model.json' in pos_json
                  and pos_json.startswith(corpus_select)]
    print(json_files)
    all_d = {}
    for file_name in json_files:
        full_path = os.path.join(path_to_json, file_name)
        with open(full_path) as json_file:
            all_d[file_name.replace('.json', '')] = json.load(json_file)
    analyze_word_distribution(all_d, absoulte=False)
    analyze_word_distribution(all_d)
