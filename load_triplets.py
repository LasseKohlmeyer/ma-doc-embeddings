from collections import defaultdict
import pandas as pd

if __name__ == "__main__":
    ids = defaultdict(list)
    file_name = 'book_comparison_data/book_rel_r.csv'
    with open(file_name, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        for j, cell in enumerate(line.split(';')):
            if cell == "1" or cell == "-1":
                ids[i + 1].append(j + 1)

    print(ids)
    triplets = []
    for a, values in ids.items():
        for i, b in enumerate(values):
            for j, c in enumerate(values):
                if i >= j:
                    continue
                triplets.append((a, b, c))

    # for triplet in triplets:
        # print(triplet)
        # a, b, c = triplet
        # print(f'True? {a} more similar to {b} for time as {c}')
    print(len(triplets))

    d = defaultdict(list)
    for (a, b, c) in triplets:
        d[a].append(1)
        d[b].append(1)
        d[c].append(1)
    d = {key: sum(values) for key, values in d.items()}
    print(d)

    df = pd.DataFrame(triplets, columns=["A", "B", "C"])
    df.to_csv('book_comparison_data/triplets.csv', index=False)

    for x in range(1,21):
        print(d[x])
