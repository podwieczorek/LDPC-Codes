import argparse
import os
import glob
import numpy as np


def convert_array_to_txt(arr, path, alist_file_path):
    alist_file_name = alist_file_path.split('/')[-1]
    txt_file_name = alist_file_name.replace('alist', 'txt')
    file_path = os.path.join(path, txt_file_name)
    try:
        np.savetxt(file_path, arr, fmt='%d', delimiter=' ')
    except Exception as e:
        print(f'{e} could not save file: {file_path}! Check if file already exists')


def convert_alist_to_array(alist_content):
    try:
        # first line contains matrix dimensions
        num_of_columns, num_of_rows = alist_content[0].strip().split()
        parity_check_matrix = np.zeros((int(num_of_rows), int(num_of_columns)))
        # entries of the columns start in the 4th row
        start_index, end_index = 4, 4 + int(num_of_columns)
        for row_number, column_entry in enumerate(alist_content[start_index:end_index]):
            # every column entry contains positions of '1's in that column
            ones_positions = column_entry.strip().split()
            for one_index in ones_positions:
                if one_index != '0':
                    parity_check_matrix[int(one_index)-1][row_number] = 1
                else:
                    # entries are right padded with zeros
                    break
        return parity_check_matrix
    except Exception as e:
        print(f'{e}, could not convert file. Check file format!')
        return None


def convert_alist_to_txt(path):
    # create directory for .txt files
    txt_dir = os.path.join(path, 'HMatricesTxt')
    if not os.path.isdir(txt_dir):
        os.mkdir(txt_dir)

    # convert all .alist files to .txt files and save it in created  directory
    for alist_file in glob.glob(os.path.join(path, '*.alist')):
        with open(alist_file, 'r') as file:
            content = file.readlines()
            parity_check_matrix = convert_alist_to_array(content)
            if parity_check_matrix is not None:
                convert_array_to_txt(parity_check_matrix, txt_dir, alist_file)


def main():
    parser = argparse.ArgumentParser(description='Script converts all parity check matrices in given directory'
                                                 'from .alist to .txt format')
    parser.add_argument('-p', '--path', help='Path to the directory with .alist files', required=True)
    args = vars(parser.parse_args())

    # check if given directory exists
    if not os.path.isdir(args['path']):
        print(f"The directory {args['path']} does not exist.")
        return

    convert_alist_to_txt(args['path'])


if __name__ == "__main__":
    main()
