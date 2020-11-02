import json
import os
from pathlib import Path


def replace_symlink_json_files(dataset_dir):
    filename = dataset_dir.name
    labels_dictionary = {}
    for class_label in list(sorted(dataset_dir.glob('*'))):
        labels_dictionary[class_label.name] = []

        for sym_link in list(sorted(class_label.glob('*'))):
            if str(sym_link).startswith(r'/data/p288722/dresden/source_devices/natural_patches'):
                sym_link = r'/data/p288722/dresden/source_devices/nat_patches_128x128_20' + sym_link[52:]
            if Path(sym_link).is_symlink():
                img_path = os.readlink(sym_link)
            else:
                img_path = sym_link
            while img_path != sym_link:
                sym_link = img_path
                if sym_link.startswith(r'/data/p288722/dresden/source_devices/natural_patches'):
                    sym_link = r'/data/p288722/dresden/source_devices/nat_patches_128x128_20' + sym_link[52:]
                if Path(sym_link).is_symlink():
                    img_path = os.readlink(sym_link)
                else:
                    img_path = sym_link
            labels_dictionary[class_label.name].append(str(img_path).strip())

    json_dictionary = {'file_paths': labels_dictionary}

    json_file_path = dataset_dir.parent.joinpath(f'{filename}.json')
    with open(json_file_path, 'w+') as f:
        json_string = json.dumps(json_dictionary, indent=2)
        f.write(json_string)


def convert_txt_to_json(text_file_path):
    with open(text_file_path, 'r') as f:
        img_paths = f.readlines()

    labels_dictionary = {}
    for path in img_paths:
        path = Path(path)
        class_label = path.parent.name
        if class_label not in labels_dictionary:
            labels_dictionary[class_label] = [str(path).strip()]
        else:
            labels_dictionary[class_label].append(str(path).strip())

    json_dictionary = {'file_paths': labels_dictionary}

    json_file_path = text_file_path.parent.joinpath(f'{text_file_path.stem}.json')
    with open(json_file_path, 'w+') as f:
        json_string = json.dumps(json_dictionary, indent=2)
        f.write(json_string)


if __name__ == '__main__':
    import shutil
    for item in Path(rf'/data/p288722/dresden/train').glob('natural'):
        if item.is_dir():
            replace_symlink_json_files(dataset_dir=item)
            # shutil.rmtree(item)

    for item in Path(rf'/data/p288722/dresden/test').glob('natural'):
        if item.is_dir():
            replace_symlink_json_files(dataset_dir=item)
            # shutil.rmtree(item)
        # convert_txt_to_json(item)
