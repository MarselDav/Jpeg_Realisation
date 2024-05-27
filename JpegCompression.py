import numpy as np
from PIL import Image

from BitsBuffer import BitsBuffer
from ColorSchemeConverter import ColorSchemeConverter, ColorSchemes
from DiscreteCosinTransformation import DiscreteCosinTransformation
from InverseBits import inverse_bits
from Quantization import Quantization, QuantizationMatrixType
from ReSample import ReSample, DownSampleType
from StatisticalTables import *
from HuffmanTablesManager import HuffmanTablesManager
from QuantizationTableIO import QuantizationTableIO


class JPEG_Compression:
    def __init__(self, quality):
        self.color_converter = ColorSchemeConverter()
        self.dct = DiscreteCosinTransformation()
        self.quantization = Quantization()
        self.resample = ReSample()
        self.bits_buffer = BitsBuffer()


        self.BYTES_CNT_TO_WRITE_RESOLUTION = 2
        self.BYTES_CNT_TO_WRITE_DOWNSAMPLE_COEF = 1

        self.QUALITY = quality

        self.CB_DOWNSAMPLING_COEFF = 2
        self.CR_DOWNSAMPLING_COEFF = 2

        self.CATEGORIES_CNT = 11
        self.MAX_ZEROS_SEQUENCE_LEN = 16
        self.BLOCK_ROWS = 8
        self.BLOCK_COLUMNS = 8

        self.AC_CNT = self.BLOCK_ROWS * self.BLOCK_COLUMNS - 1

        self.EOB = (0, 0)
        self.ZRL = (15, 0)


    def process_blocks(self, component, quantizationMatrixType):
        height, width = component.shape

        height -= height % self.BLOCK_ROWS
        width -= width % self.BLOCK_COLUMNS

        quantized_matricies = []

        for y in range(0, height, self.BLOCK_ROWS):
            for x in range(0, width, self.BLOCK_COLUMNS):
                matrix_8x8 = component[y:y + self.BLOCK_ROWS, x:x + self.BLOCK_COLUMNS]
                matrix_8x8_shift = matrix_8x8
                fdct_matrix = self.dct.forward_dct(matrix_8x8_shift)
                quantized_matrix = self.quantization.quantization(fdct_matrix, quantizationMatrixType)
                quantized_matricies.append(quantized_matrix)

        return quantized_matricies

    def zigzag_conversion(self, matrix: np.array, result_list: list[int]):
        x, y = 1, 0

        while True:
            result_list.append(matrix[x, y])

            if x == self.BLOCK_COLUMNS - 1 and y == self.BLOCK_ROWS - 1:
                break

            if (x + y) % 2 == 0:  # Движение вправо-вверх
                if y == 0:
                    x += 1
                elif x == self.BLOCK_COLUMNS - 1:
                    y += 1
                else:
                    x += 1
                    y -= 1
            else:
                if y == self.BLOCK_ROWS - 1:
                    x += 1
                elif x == 0:  # Движение влево-вниз
                    y += 1
                else:
                    x -= 1
                    y += 1

    def zigzag_conversion_matricies(self, matricies):
        DC_list = []
        AC_list = []

        for matrix in matricies:
            DC_list.append(matrix[0, 0])
            self.zigzag_conversion(matrix, AC_list)

        return DC_list, AC_list

    @staticmethod
    def get_category_and_code(num):
        category = int(np.ceil(np.log2(abs(num) + 1)))

        encoded_dc = bin(num)[2::]
        if num < 0:
            encoded_dc = inverse_bits(encoded_dc[1::])

        return category, encoded_dc

    def encode_dc(self, dc_list: bytearray, category_huffman_table):
        prev = 0
        for dc in dc_list:
            deltaDC = dc - prev
            prev = dc

            category, encoded_dc = self.get_category_and_code(deltaDC)

            b = [category_huffman_table[category]]

            self.bits_buffer.add(category_huffman_table[category])
            if category != 0:
                self.bits_buffer.add(encoded_dc)
                b.append(encoded_dc)

        self.bits_buffer.write()

    def encode_ac(self, ac_list, ac_huffman_table):
        current_blocK_size = 0

        rle_len = 0

        run_length = 0
        for ac in ac_list:
            if current_blocK_size == self.AC_CNT:
                if run_length > 0:
                    self.bits_buffer.add(ac_huffman_table[self.EOB])
                    rle_len += 2
                    run_length = 0
                current_blocK_size = 0

            if ac:
                while run_length >= self.MAX_ZEROS_SEQUENCE_LEN:
                    self.bits_buffer.add(ac_huffman_table[self.ZRL])
                    rle_len += 2
                    run_length -= self.MAX_ZEROS_SEQUENCE_LEN

                size, encoded_ac = self.get_category_and_code(ac)
                self.bits_buffer.add(ac_huffman_table[(run_length, size)])
                rle_len += 2
                self.bits_buffer.add(encoded_ac)

                run_length = 0
            else:
                run_length += 1

            current_blocK_size += 1

        if run_length > 0:
            rle_len += 2
            self.bits_buffer.add(ac_huffman_table[self.EOB])

        # print("rle_len: ", rle_len)
        self.bits_buffer.write()

    def encode_component(self, componet, dc_table, ac_table):
        dc_list, ac_list = self.zigzag_conversion_matricies(componet)
        # print("dc len: ", len(dc_list))
        self.encode_dc(dc_list, dc_table)
        self.encode_ac(ac_list, ac_table)

    def compression(self, file_path):
        image_array = np.array(Image.open(file_path))

        new_file_name = file_path.split(".")[0] + "_jpeg.txt"

        with open(new_file_name, "wb") as file_writer:
            self.bits_buffer.set_file_writer(file_writer)
            Y, Cb, Cr = self.color_converter.convert(image_array, ColorSchemes.YCbCr)

            original_size = image_array.shape[:2:]
            file_writer.write(original_size[0].to_bytes(self.BYTES_CNT_TO_WRITE_RESOLUTION, byteorder="big"))
            file_writer.write(original_size[1].to_bytes(self.BYTES_CNT_TO_WRITE_RESOLUTION, byteorder="big"))

            file_writer.write(self.CB_DOWNSAMPLING_COEFF.to_bytes(self.BYTES_CNT_TO_WRITE_DOWNSAMPLE_COEF, byteorder="big"))
            file_writer.write(self.CR_DOWNSAMPLING_COEFF.to_bytes(self.BYTES_CNT_TO_WRITE_DOWNSAMPLE_COEF, byteorder="big"))

            QuantizationTableIO.write_quality_and_quantization_matrices(file_writer, self.QUALITY, self.quantization)
            self.quantization.scale_quantization_matrix(self.QUALITY)

            HuffmanTablesManager.write_dc_table(file_writer, HUFFMAN_DC_LUMINANCE_TABLE_FORWARD)
            HuffmanTablesManager.write_dc_table(file_writer, HUFFMAN_DC_CHROMINANCE_TABLE_FORWARD)
            HuffmanTablesManager.write_ac_table(file_writer, HUFFMAN_AC_LUMINANCE_TABLE_FORWARD)
            HuffmanTablesManager.write_ac_table(file_writer, HUFFMAN_AC_CHROMINANCE_TABLE_FORWARD)

            # downsampling Cb и Cr
            Cb = self.resample.downSample(Cb, self.CB_DOWNSAMPLING_COEFF, self.CB_DOWNSAMPLING_COEFF, DownSampleType.RemovePixels)
            Cr = self.resample.downSample(Cr, self.CR_DOWNSAMPLING_COEFF, self.CR_DOWNSAMPLING_COEFF, DownSampleType.RemovePixels)

            Y_matricies = self.process_blocks(Y, QuantizationMatrixType.L)
            Cb_matricies = self.process_blocks(Cb, QuantizationMatrixType.C)
            Cr_matricies = self.process_blocks(Cr, QuantizationMatrixType.C)

            self.encode_component(Y_matricies, HUFFMAN_DC_LUMINANCE_TABLE_FORWARD, HUFFMAN_AC_LUMINANCE_TABLE_FORWARD)
            self.encode_component(Cb_matricies, HUFFMAN_DC_CHROMINANCE_TABLE_FORWARD,
                                  HUFFMAN_AC_CHROMINANCE_TABLE_FORWARD)
            self.encode_component(Cr_matricies, HUFFMAN_DC_CHROMINANCE_TABLE_FORWARD,
                                  HUFFMAN_AC_CHROMINANCE_TABLE_FORWARD)


if __name__ == "__main__":
    # original_image = "images/Lenna.png"
    original_image = "images/Lenna512.png"
    # original_image = "images/small_image_50x50.jpg"
    jpeg_compression = JPEG_Compression(50)
    jpeg_compression.compression(original_image)
