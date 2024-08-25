import argparse
import os
import warnings


def convert_to_array(file, path):
    file_path = os.path.join(path, 'converted_to_array.txt')
    content = file.readlines()
    with open(file_path, 'w') as new_file:
        row_number = len(content)
        new_file.write('[')
        for row_index, row in enumerate(content):
            new_file.write('[')
            new_file.write(row.replace(' ', ', '))
            new_file.write(']')
            if row_index != row_number-1:
                new_file.write(', ')
        new_file.write(']')


def main():
    parser = argparse.ArgumentParser(description='Program converts parity check matrices in .txt format'
                                                 'to array format that can be used in GAP')
    parser.add_argument('-p', '--path', help='Path to the .txt file', required=True)
    path = vars(parser.parse_args())['path']

    # check if given file exists
    if not os.path.isfile(path):
        warnings.warn(f"File {path} does not exist, check the path!")
        return

    with open(path, 'r') as file:
        convert_to_array(file, os.path.dirname(path))


if __name__ == "__main__":
    main()
