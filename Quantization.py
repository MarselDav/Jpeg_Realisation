import numpy as np


class QuantizationMatrixType:
    L = 1
    C = 2


class Quantization:
    def __init__(self):
        self.scaled_luminance_matrix = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]])

        self.scaled_chrominance_matrix = np.array([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]])

    def set_luminance_matrix(self, matrix):
        self.scaled_luminance_matrix = matrix

    def set_chrominance_matrix(self, matrix):
        self.scaled_chrominance_matrix = matrix

    def scale_quantization_matrix(self, quality):
        if quality < 50:
            s = 500 / quality
        else:
            s = 200 - 2 * quality

        self.scaled_luminance_matrix = np.clip(np.floor((self.scaled_luminance_matrix * s + 50) / 100), 1, None)
        self.scaled_chrominance_matrix = np.clip(np.floor((self.scaled_chrominance_matrix * s + 50) / 100), 1, None)

    def quantization(self, dct_matrix, type):
        if type == QuantizationMatrixType.L:
            return np.int16(dct_matrix / self.scaled_luminance_matrix)
        return np.int16(dct_matrix / self.scaled_chrominance_matrix)

    def dequantization(self, dct_matrix, type):
        if type == QuantizationMatrixType.L:
            return np.int16(dct_matrix * self.scaled_luminance_matrix)
        return np.int16(dct_matrix * self.scaled_chrominance_matrix)


if __name__ == "__main__":
    q = Quantization()

    matrix = np.random.randint(0, 255, size=(8, 8))
    # print(matrix)
    q_matrix = q.quantization(matrix, QuantizationMatrixType.L)
    print(q_matrix)
    # orig_matrix = q.dequantization(q_matrix, QuantizationMatrixType.L, 100)
    # print(orig_matrix)
    # quant = q.get_scaled_quantization_matrix(QuantizationMatrixType.L, 100)
    # print(quant)
