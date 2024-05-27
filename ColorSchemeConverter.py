import numpy as np


class ColorSchemes:
    RGB = 1
    YCbCr = 2


class ColorSchemeConverter:
    def convert(self, image_matrix, color_scheme: ColorSchemes):
        if color_scheme == ColorSchemes.RGB:
            return self.__convert_YCbCr_to_RGB(image_matrix)
        elif color_scheme == ColorSchemes.YCbCr:
            return self.__convert_RGB_to_YCbCr(image_matrix)

    @staticmethod
    def __convert_YCbCr_to_RGB(ycbcr_image_matrix: np.array) -> (np.array, np.array, np.array):
        Y, Cb, Cr = ycbcr_image_matrix[:, :, 0], ycbcr_image_matrix[:, :, 1], ycbcr_image_matrix[:, :, 2]

        R = Y + 1.402 * (Cr - 128.)
        G = Y - 0.3441 * (Cb - 128.) - 0.7141 * (Cr - 128.)
        B = Y + 1.772 * (Cb - 128.)

        R = np.clip(np.round(R), 0, 255).astype(np.uint8)
        G = np.clip(np.round(G), 0, 255).astype(np.uint8)
        B = np.clip(np.round(B), 0, 255).astype(np.uint8)

        return np.dstack((R, G, B)).astype(np.uint8)
        # return R, G, B

    @staticmethod
    def __convert_RGB_to_YCbCr(rgb_image_matrix: np.array) -> (np.array, np.array, np.array):
        R, G, B = rgb_image_matrix[:, :, 0], rgb_image_matrix[:, :, 1], rgb_image_matrix[:, :, 2]

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128.
        Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128.

        Y = np.clip(np.round(Y), 0, 255).astype(np.uint8)
        Cb = np.clip(np.round(Cb), 0, 255).astype(np.uint8)
        Cr = np.clip(np.round(Cr), 0, 255).astype(np.uint8)

        # return np.dstack((Y, Cb, Cr)).astype(np.uint8)
        return Y, Cb, Cr
