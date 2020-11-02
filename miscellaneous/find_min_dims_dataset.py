from pathlib import Path

from skimage import io


# fixme: modify to read from json files
def find_min_dims_dataset(dataset_root):
    img_paths = Path(dataset_root).glob('*/*.jpg')
    min_shape = [3, 10000, 10000]
    for img_path in img_paths:
        img = io.imread(str(img_path)).transpose((2, 0, 1))  # HxWxC --> CxHxW
        min_shape = [min(a, b) for a, b in zip(min_shape, img.shape)]

    return min_shape


if __name__ == '__main__':
    dims = find_min_dims_dataset(dataset_root=r'D:\Data\Dresden\source_devices\natural')
    print(dims)
