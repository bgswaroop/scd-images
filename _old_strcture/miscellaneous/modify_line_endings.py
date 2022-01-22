import os
from pathlib import Path


def modify_line_endings(root_dir):
    # r=root, d=directories, f = files
    for r, d, f in os.walk(root_dir):
        for file in f:
            if file.endswith(".py") or file.endswith(".gitignore") or file.endswith(".sh") or file.endswith(".txt"):
                with open(file, 'r') as fp:
                    lines = fp.readlines()
                with open(file, 'w+') as fp:
                    lines = [x.replace('\r\n', '\n') for x in lines]
                    fp.writelines(lines)


if __name__ == '__main__':
    modify_line_endings(Path(rf'D:\GitCode\scd_images'))
    print('Finished')
