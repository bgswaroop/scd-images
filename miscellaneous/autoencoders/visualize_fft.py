import random

import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def fftshift(real, imag, dims=None):
    if dims is None:
        dims = range(1, len(real.size()))
    for dim in dims:
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return real, imag


def fft_abs(rgb_img, eps=1e-10):
    # The FFT of a real signal is Hermitian-symmetric, X[i, j] = conj(X[-i, -j]), so the full fft2() output contains
    # redundant information. rfft2() instead omits the negative frequencies in the last dimension. (See below)
    # x = torch.fft.fft2(rgb_img)
    # real, imag = fftshift(real=x.real, imag=x.imag)

    x = torch.fft.rfft2(rgb_img)
    real, imag = fftshift(real=x.real, imag=x.imag, dims=[1])

    x = torch.sqrt(torch.square(real) + torch.square(imag))  # magnitude of complex number
    x = torch.log(x + eps)  # scale the values, adding eps for numerical stability
    mag = (x - torch.min(x)) / (torch.max(x) - torch.min(x))  # normalize the values
    phase = torch.atan(imag / real)

    return mag, phase


def read_and_crop(img_path):
    x = Image.open(img_path)  # pillow reads the image with the  sequence of RGB (unlike openCV)
    x = torchvision.transforms.CenterCrop((480, 639))(x)
    x = torchvision.transforms.ToTensor()(x)
    return x


def visualize_ffts(fft, avg_fft):
    """
    :param fft:
    :param avg_fft:
    :return:
    """

    # Plot average fft
    plt.figure()
    fig, axarr = plt.subplots(nrows=2, ncols=9, figsize=(18, 5))
    font_dict = {'fontsize': 16, 'fontweight': 'medium'}

    # plot row 1
    for row in range(2):
        for col in range(9):
            axarr[row][col].imshow(avg_fft[row * 9 + col])
            axarr[row][col].set_title(f'Class - {row * 9 + col + 1}', fontdict=font_dict)

    plt.suptitle('Average Fourier Transform', fontsize=16)
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.cla()
    plt.close()

    # plot sample ffts for each class
    for class_idx in range(18):
        plt.figure()
        fig, axarr = plt.subplots(nrows=3, ncols=9, figsize=(18, 7))

        for row in range(3):
            for col in range(9):
                axarr[row][col].imshow(fft[class_idx][random.sample(range(len(fft[class_idx])), k=1)[0]])
        plt.suptitle(f'Sample Magnitude Maps - Class {class_idx + 1}', fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()


if __name__ == '__main__':
    img = read_and_crop(img_path=r'/data/p288722/dresden/source_devices/natural/Agfa_DC-504_0/Agfa_DC-504_0_10.JPG')
    fft, phase = fft_abs(rgb_img=img)
    fft_gray, phase_gray = fft_abs(rgb_img=torch.mean(img, dim=0, keepdim=True))

    # Set-up the figure
    plt.figure()
    fig, axarr = plt.subplots(nrows=3, ncols=3, figsize=(12, 11))
    font_dict = {'fontsize': 16, 'fontweight': 'medium'}

    # plot row 1
    axarr[0][0].imshow(img.permute(1, 2, 0))
    axarr[0][0].set_title('Original Image', fontdict=font_dict)
    axarr[0][1].imshow(fft_gray[0])
    axarr[0][1].set_title('log_mag - gray', fontdict=font_dict)
    axarr[0][2].imshow(phase_gray[0])
    axarr[0][2].set_title('phase - gray', fontdict=font_dict)

    # plot row 2
    for i, title in enumerate(['log_mag - red', 'log_mag - green', 'log_mag - blue']):
        axarr[1][i].imshow(fft[i])
        axarr[1][i].set_title(title, fontdict=font_dict)

    # plot row 3
    for i, title in enumerate(['phase - red', 'phase - green', 'phase - blue']):
        axarr[2][i].imshow(phase[i])
        axarr[2][i].set_title(title, fontdict=font_dict)

    plt.suptitle('Fourier transforms', fontsize=16)
    plt.tight_layout()
    plt.show()

    plt.clf()
    plt.cla()
    plt.close()

    print()
    # plot fft
