import numpy as np
from PIL import Image


def save_mean_sub_image(filename):
    img = np.array(Image.open(filename))
    img = img / 255.0
    scaled_img = img - np.mean(img, axis=(1, 2), keepdims=True)
    scaled_img = (scaled_img + 1) / 2.0
    scaled_img = (scaled_img * 255.0).astype(np.uint8)
    scaled_img = Image.fromarray(scaled_img)
    scaled_img.save('scaled_' + filename)
    print(' ')


if __name__ == '__main__':
    for image in [
        'patch_S00247_R0068_G0008_B0099.png',
        'patch_S00511_R0125_G0174_B0212.png',
        'patch_S00555_R0186_G0186_B0183.png',
    ]:
        save_mean_sub_image(image)
