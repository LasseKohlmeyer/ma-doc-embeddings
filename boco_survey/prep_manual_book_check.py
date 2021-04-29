import pandas as pd
import os

books_mapping = [["Uncle Tom's Cabin by Harriet Beecher Stowe", "Onkel Toms Hütte von Harriet Beecher Stowe",
                  '01'],
                 ['A Tale of Two Cities by Charles Dickens', 'Eine Geschichte aus zwei Städten von Charles Dickens',
                  '02'],
                 ['Adventures of Huckleberry Finn by Mark Twain', 'Die Abenteuer des Huckleberry Finn von Mark Twain',
                  '03'],
                 ['Alice’s Adventures in Wonderland by Lewis Carroll', 'Alice im Wunderland von Lewis Carroll',
                  '04'],
                 ['Dracula by Bram Stoker', 'Dracula von Bram Stoker',
                  '05'],
                 ['Emma by Jane Austen', 'Emma von Jane Austen',
                  '06'],
                 ['Frankenstein by Mary Shelley', 'Frankenstein; oder: Der moderne Prometheus von Mary Shelley',
                  '07'],
                 ['Great Expectations by Charles Dickens', 'Große Erwartungen von Charles Dickens',
                  '08'],
                 ['Metamorphosis by Franz Kafka', 'Die Verwandlung von Franz Kafka',
                  '09'],
                 ['Pride and Prejudice by Jane Austen', 'Stolz und Vorurteil von Jane Austen',
                  '10'],
                 ['The Adventures of Sherlock Holmes by Arthur C. Doyle',
                  'Die Abenteuer des Sherlock Holmes von Arthur C. Doyle',
                  '11'],
                 ['The Adventures of Tom Sawyer by Mark Twain', 'Die Abenteuer des Tom Sawyer von Mark Twain',
                  '12'],
                 ['The Count of Monte Cristo by Alexandre Dumas', 'Der Graf von Monte Christo von Alexandre Dumas',
                  '13'],
                 ['The Picture of Dorian Gray by Oscar Wilde', 'Das Bildnis des Dorian Gray von Oscar Wilde',
                  '14'],
                 ['Little Women by Louisa M. Alcott', 'Little Women von Louisa M. Alcott',
                  '15'],
                 ['Heart of Darkness by Joseph Conrad', 'Herz der Finsternis von Joseph Conrad',
                  '16'],
                 ['Moby Dick by Herman Melville', 'Moby-Dick; oder: Der Wal von Herman Melville',
                  '17'],
                 ['War and Peace by Leo Tolstoy', 'Krieg und Frieden von Leo Tolstoy',
                  '18'],
                 ['Wuthering Heights by Emily Brontë', 'Sturmhöhe von Emily Brontë',
                  '19'],
                 ['Treasure Island by Robert L. Stevenson', 'Die Schatzinsel von Robert L. Stevenson',
                  '20']]

books_mapping = {int(b_id): eng for (eng, de, b_id) in books_mapping}

df = pd.read_csv("../results/human_assessment/gutenberg_classic_20/human_assessed.csv")

if os.path.isfile("../results/human_assessment/gutenberg_classic_20/self_assessed_complete.csv"):
    df_comp = pd.read_csv("../results/human_assessment/gutenberg_classic_20/self_assessed_complete.csv")
    print(df_comp)

    new_assessed = {}
    for i, row in df_comp.iterrows():
        new_assessed[(row["Book 1"],
                      row["Book 2"],
                      row["Book 3"],
                      row["Facet"])] = (row["Selection"], row["Selected Answer Nr."])

    merged_tuples = []
    for i, row in df.iterrows():
        selection = row["Selection"]
        answer_nr = row["Selected Answer Nr."]
        agreement = row["Agreement"]
        answers = int(row["Answers"])

        if agreement <= 0.5:
            selection, answer_nr = new_assessed[(row["Book 1"], row["Book 2"], row["Book 3"], row["Facet"])]
            answers += 1

        merged_tuples.append((row["Book 1"], row["Book 2"], row["Book 3"], row["Facet"], selection,
                              answer_nr, agreement, answers))

    new_df = pd.DataFrame(merged_tuples, columns=["Book 1", "Book 2", "Book 3", "Facet", "Selection",
                                                  "Selected Answer Nr.", "Agreement", "Answers"])

    new_df.to_csv("results/human_assessment/gutenberg_classic_20/human_assessed_complete.csv", index=False)


else:
    df = df.loc[df['Agreement'] <= 0.5]
    df = df[["Book 1", "Book 2", "Book 3", "Facet", "Selection", "Selected Answer Nr.", "Answers"]]

    print(df)

    df["Title 1"] = df["Book 1"].map(books_mapping)
    df["Title 2"] = df["Book 2"].map(books_mapping)
    df["Title 3"] = df["Book 3"].map(books_mapping)
    print(df)

    df.to_csv("results/human_assessment/gutenberg_classic_20/self_assessed.csv", index=False)