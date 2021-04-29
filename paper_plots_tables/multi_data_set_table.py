import os
import pandas as pd
input_folder = "result_series_multi_dataset"

df = pd.read_csv(os.path.join(input_folder, "z_table_genre.csv"))

metric = "ndcg"
df = df[["Dataset", "Task", "Algorithm", metric]]
df[metric] = df[metric].apply(lambda x: float(str(x).split(" ± ")[0]))
df = df.pivot(index=["Task", "Algorithm"], columns="Dataset", values=metric)

print(df)
print(df.to_latex())