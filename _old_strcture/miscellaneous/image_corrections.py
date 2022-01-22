import cv2
from pathlib import Path
import numpy as  np
from matplotlib import pyplot as plt
from skimage import exposure


def plot_transformations(original, hist_equalized, gamma_encoded, gamma_decoded):
    plt.figure()
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))

    axarr[0][0].imshow(original)
    axarr[0][1].imshow(hist_equalized)
    axarr[1][0].imshow(gamma_encoded)
    axarr[1][1].imshow(gamma_decoded)

    axarr[0][0].set_title('Original Image', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    axarr[0][1].set_title('Histogram Equalized', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    axarr[1][0].set_title('Gamma Encoding (factor of 2.2)', fontdict={'fontsize': 12, 'fontweight': 'medium'})
    axarr[1][1].set_title('Gamma Decoding (factor of 2.2)', fontdict={'fontsize': 12, 'fontweight': 'medium'})

    plt.suptitle('Image transformations')
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def hist_eq(img_input):
    # Convert the RGB image inti YUV space
    img_hsv = cv2.cvtColor(img_input, cv2.COLOR_RGB2HSV)

    # equalize the histogram of the Y channel
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    return img_output


gamma = 2.2
inverse_gamma = 1.0 / gamma
encoding_table = np.array([((i / 255.0) ** inverse_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
decoding_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")


def gamma_encoding(image):
    return cv2.LUT(image, encoding_table)


def gamma_decoding(image):
    return cv2.LUT(image, decoding_table)


if __name__ == '__main__':
    image_path = Path(r'/data/p288722/dresden/source_devices/natural/Canon_Ixus70_0/Canon_Ixus70_0_3283.JPG')
    img_bgr = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # img = np.float32(img) / 255.0

    plot_transformations(
        original=img_rgb,
        hist_equalized=hist_eq(img_rgb),
        gamma_encoded=gamma_encoding(img_rgb),
        gamma_decoded=gamma_decoding(img_rgb)
    )
