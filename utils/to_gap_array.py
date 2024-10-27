import argparse
import os
import glob
import warnings


def convert_to_array(path):
    # create directory for converted files
    new_dir = os.path.join(path, 'GapFormat')
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)

    # convert all .alist files to .txt files and save it in created  directory
    for txt_file in glob.glob(os.path.join(path, '*.txt')):
        with open(txt_file, 'r') as file:
            file_name = txt_file.split('/')[-1][:-4] + '_gap.txt'
            new_file_path = os.path.join(new_dir, file_name)
            convert_one_file_to_array(file, new_file_path)


def convert_one_file_to_array(file, new_file_path):
    if os.path.isdir(new_file_path):
        new_file_path = os.path.join(new_file_path, 'h_gap.txt')

    content = file.readlines()
    with open(new_file_path, 'w') as new_file:
        row_number = len(content)
        new_file.write('[')
        for row_index, row in enumerate(content):
            new_file.write('[')
            new_file.write(row.replace(' ', ', '))
            new_file.write(']')
            if row_index != row_number - 1:
                new_file.write(', ')
        new_file.write(']')


def main():
    parser = argparse.ArgumentParser(description='Program converts parity check matrices in .txt format'
                                                 'to format that can be used in GAP')
    parser.add_argument('-p', '--path', help='Path to the .txt file', required=True)
    path = vars(parser.parse_args())['path']

    if os.path.isfile(path):
        with open(path, 'r') as file:
            convert_one_file_to_array(file, os.path.dirname(path))
    elif os.path.isdir(path):
        convert_to_array(path)
    else:
        warnings.warn(f"{path} does not exist, check if directory or file exists!")
        return


if __name__ == "__main__":
    main()
