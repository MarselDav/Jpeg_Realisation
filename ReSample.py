from math import ceil

import numpy as np


class DownSampleType:
    RemovePixels = 1
    ReplaceByAverageColor = 2
    ReplaceByApproxColor = 3


class ReSample:
    def downSample(self, matrix, Cx, Cy, type: DownSampleType):
        if type == DownSampleType.RemovePixels:
            return self.__downSampleRemovePixels(matrix, Cx, Cy)

        elif type == DownSampleType.ReplaceByAverageColor:
            return self.__downSampleReplaceByAverageColor(matrix, Cx, Cy)

        elif type == DownSampleType.ReplaceByApproxColor:
            return self.__downSampleReplaceByApproxColor(matrix, Cx, Cy)

    @staticmethod
    def upSample(image_matrix, original_shape: np.shape):
        N, M = image_matrix.shape
        Orig_N, Orig_M = original_shape
        Cy, Cx = ceil(Orig_N / N), ceil(Orig_M / M)

        image_matrix = np.repeat(np.repeat(image_matrix, Cy, axis=0), Cx, axis=1)

        return image_matrix[:Orig_N, :Orig_M]

    @staticmethod
    def __downSampleRemovePixels(image_matrix: np.array, Cx, Cy) -> np.array:
        N, M = image_matrix.shape
        New_N, New_M = ceil(N / Cy), ceil(M / Cx)
        new_image_matrix = np.empty((New_N, New_M), dtype=np.uint8)

        new_image_matrix[:, :] = image_matrix[::Cy, ::Cx]

        return new_image_matrix

    @staticmethod
    def __downSampleReplaceByAverageColor(image_matrix: np.array, Cx, Cy) -> np.array:
        N, M, _ = image_matrix.shape
        New_N, New_M = ceil(N / Cy), ceil(M / Cx)
        new_image_matrix = np.empty((New_N, New_M, 3), dtype=np.uint8)

        for y in range(New_N):
            for x in range(New_M):
                # Определение границ для подматрицы
                x_start = x * Cx
                x_end = min(x_start + Cx, M)

                y_start = y * Cy
                y_end = min(y_start + Cy, N)

                # Вычисление среднего цвета
                sub_matrix = image_matrix[y_start:y_end, x_start:x_end]
                average_color = np.mean(sub_matrix, axis=(0, 1))

                new_image_matrix[y, x] = average_color.astype(np.uint8)

        return new_image_matrix

    @staticmethod
    def __downSampleReplaceByApproxColor(image_matrix: np.array, Cx, Cy) -> np.array:
        N, M = image_matrix.shape
        New_N, New_M = ceil(N / Cy), ceil(M / Cx)
        new_image_matrix = np.empty((New_N, New_M), dtype=np.uint8)

        for y in range(New_N):
            for x in range(New_M):
                # Определение границ для подматрицы
                x_start = x * Cx
                x_end = min(x_start + Cx, M)

                y_start = y * Cy
                y_end = min(y_start + Cy, N)

                sub_matrix = image_matrix[y_start:y_end, x_start:x_end]
                average_color = np.mean(sub_matrix, axis=(0, 1))  # Вычисление среднего цвета

                abs_difference_sub_matrix = np.abs(
                    sub_matrix - average_color)  # находим абсолютную разницу между каждым пикседем и средним

                index = np.argmin(np.sum(abs_difference_sub_matrix, axis=1))  # индекс цвета с минимальной разницей
                # # unravel_index - преобраование одномерного индекса в многомерный
                color_y, color_x = np.unravel_index(index, sub_matrix.shape[:2:])
                closest_color = sub_matrix[color_y, color_x]
                new_image_matrix[y, x] = closest_color

        return new_image_matrix
