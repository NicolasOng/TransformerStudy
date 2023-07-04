# generated using ChatGPT

import os

def concat_md_files(root_dir, output_file):
    with open(output_file, 'w') as outfile:
        for dir_path, dir_names, file_names in os.walk(root_dir):
            for file_name in file_names:
                if file_name.endswith('.md'):
                    file_path = os.path.join(dir_path, file_name)
                    with open(file_path, 'r') as infile:
                        outfile.write(infile.read())
                        # add a newline character for separation between files
                        outfile.write('\n')

root_dir = 'Knowledge'  # replace with your directory path
output_file = 'output.txt'  # replace with your desired output file path

concat_md_files(root_dir, output_file)

