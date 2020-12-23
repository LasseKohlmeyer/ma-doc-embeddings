from striprtf.striprtf import rtf_to_text
import os
from os import listdir
from os.path import isfile, join


def parse_file(input_dir:str, input_file_name: str, output_dir: str):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    input_file_path = os.path.join(input_dir, input_file_name)
    output_file_path = os.path.join(output_dir, input_file_name.replace('.rtf', '.txt'))
    with open(input_file_path, 'r', encoding='ansi') as in_file:
        text = rtf_to_text(in_file.read())

        with open(output_file_path, 'w', encoding='utf-8') as out_file:
            out_file.write(text)


if __name__ == '__main__':
    path = 'E:/Corpora/guttenberg_top20popular'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for file in files:
        print(file)
        parse_file(path, file, "E:\Corpora\gutenberg_top_20_parsed")