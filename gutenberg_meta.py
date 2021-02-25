import re
import pandas as pd


def correct_gutenberg_meta(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as file:
        content = file.read()
        content = re.sub(r'\n\n+', ' ', content)
        # content = content.replace('\n\n', '')
        # print(content[:1000])

        new_lines = []
        last_line = ""
        for line_count, line in enumerate(content.split('\n')):
            line_id_cand = line.split(',')[0]
            if line_id_cand.isdigit():
                new_lines.append(last_line)
                last_line = ""
            else:
                pass

            last_line += line

        if last_line:
            new_lines.append(last_line)

        new_content = '\n'.join(new_lines)

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(new_content)


def load_gutenberg_meta(path: str):
    df = pd.read_csv(path)
    d = {}
    # gutenberg_id,title,author,gutenberg_author_id,language,gutenberg_bookshelf,rights,has_text
    for i, row in df.iterrows():
        author = row['author']
        if not pd.isna(author):
            splitted = author.split(', ')
            splitted.reverse()
            author = ' '.join(splitted)
        else:
            author = None

        d[str(row['gutenberg_id'])] = (row['title'], author, row['gutenberg_bookshelf'])

    return d


if __name__ == '__main__':
    # correct_gutenberg_meta("E:/Corpora/Gutenberg_Meta/gutenberg_metadata_old.csv",
    #                        "E:/Corpora/Gutenberg_Meta/gutenberg_metadata.csv")
    guten_dict = load_gutenberg_meta("E:/Corpora/Gutenberg_Meta/gutenberg_metadata.csv")
    print(guten_dict["23180"])
