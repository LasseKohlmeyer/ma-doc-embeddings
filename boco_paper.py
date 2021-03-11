from collections import defaultdict

import pandas as pd

df = pd.read_csv("results/human_assessment/gutenberg_classic_20/book_familarity.csv")
print(df)
df = df[["Book ID", "Description", "Language", "Unknown", "Known"]]
df = df.pivot(index=["Book ID", "Description"], columns="Language", values=["Unknown", "Known"])
# df = df.sort_values(df["Known"]["all"],ascending=False, axis=1)
# d = defaultdict(list)
# for i, row in df.iterrows():
#     d[row["Book ID"]].append((f'{row["Language"]} known Books Ratio', row["Known"] / (row["Known"] + row["Unknown"])))
# print(d)
print(df.to_latex(index=True))
