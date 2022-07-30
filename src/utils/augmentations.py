# referring to https://github.com/liuquande/FedDG-ELCFS

import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class AmplitudeInterpolation:
    def __init__(self, client_idx, global_amplitude_bank, L=0.003):
        self.client_idx = client_idx
        self.global_amplitude_bank = global_amplitude_bank
        self.L = L

    def __call__(self, img):
        img = img.transpose((2, 0, 1))
        r = list(range(0, self.client_idx)) + list(range(self.client_idx + 1, len(self.global_amplitude_bank)))
        random_client = random.choice(r)

        r = range(len(self.global_amplitude_bank[random_client][0]))
        random_target = random.choice(r)

        amp_target = np.load(self.global_amplitude_bank[random_client][0][random_target])

        # with_visualization = False
        # if with_visualization:
        #     dataset_name = self.global_amplitude_bank[random_client][1]
        #     path = self.global_amplitude_bank[random_client][0][random_target]
        #     surgery_number = path.split("/")[3]
        #     filename = path.split("\\")[-1][4:12]
        #     target = np.asarray(Image.open(f"datasets/train/{dataset_name}/{surgery_number}/{filename}.png"),
        #                         np.float32)
        #
        #     fig, ax = plt.subplots(1, 8)
        #     plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        #     x = img.transpose((1, 2, 0))
        #     ax[0].imshow(x / 255)
        #     ax[6].imshow(target / 255)
        #
        #
        #     # amp_target already transposed during preprocessing
        #     # amp_target = amp_target.transpose((2, 0, 1))
        #
        #     amp_target_shift = np.fft.fftshift(amp_target, axes=(-2, -1))
        #     ax[7].imshow(np.clip((np.log(amp_target_shift) / np.max(np.log(amp_target_shift)))
        #                          .transpose((1, 2, 0)), 0, 1), cmap="gray")
        #
        #     for i, ratio in enumerate([0.2, 0.4, 0.6, 0.8, 1.]):
        #         local_in_trg = freq_space_interpolation(img, amp_target, L=self.L, ratio=1 - ratio)
        #         local_in_trg = local_in_trg.transpose((1, 2, 0))
        #         ax[i+1].imshow(np.clip(local_in_trg / 255, 0, 1))
        #
        #     plt.show()

        ratio = np.random.uniform()
        local_in_trg = freq_space_interpolation(img, amp_target, L=self.L, ratio=1 - ratio)
        local_in_trg = local_in_trg.transpose((1, 2, 0))
        local_in_trg = np.clip(local_in_trg, 0, 255).astype(np.uint8)
        return local_in_trg


def amp_spectrum_swap(amp_local, amp_target, L=0.1, ratio=0):
    a_local = np.fft.fftshift(amp_local, axes=(-2, -1))
    a_trg = np.fft.fftshift(amp_target, axes=(-2, -1))

    _, h, w = a_local.shape
    b = (np.floor(np.amin((h, w)) * L)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    a_local[:, h1:h2, w1:w2] = a_local[:, h1:h2, w1:w2] * ratio + a_trg[:, h1:h2, w1:w2] * (1 - ratio)
    a_local = np.fft.ifftshift(a_local, axes=(-2, -1))
    return a_local


def freq_space_interpolation(local_img, amp_target, L=0, ratio=0):
    local_img_np = local_img

    # get fft of local sample
    fft_local_np = np.fft.fft2(local_img_np, axes=(-2, -1))

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    amp_local_ = amp_spectrum_swap(amp_local, amp_target, L=L, ratio=ratio)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp(1j * pha_local)
    local_in_trg = np.fft.ifft2(fft_local_, axes=(-2, -1))
    local_in_trg = np.real(local_in_trg)

    return local_in_trg
