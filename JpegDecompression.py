import numpy as np
from PIL import Image

from BitsBuffer import BitsBuffer
from ColorSchemeConverter import ColorSchemeConverter, ColorSchemes
from DiscreteCosinTransformation import DiscreteCosinTransformation
from HuffmanTablesManager import HuffmanTablesManager
from InverseBits import inverse_bits
from Quantization import Quantization, QuantizationMatrixType
from QuantizationTableIO import QuantizationTableIO
from ReSample import ReSample


class JPEG_Decompression:
    def __init__(self, quality):
        self.color_converter = ColorSchemeConverter()
        self.dct = DiscreteCosinTransformation()
        self.quantization = Quantization()
        self.resample = ReSample()
        self.bits_buffer = BitsBuffer()

        self.BYTES_CNT_TO_WRITE_RESOLUTION = 2
        self.BYTES_CNT_TO_WRITE_DOWNSAMPLE_COEF = 1

        self.QUALITY = quality

        self.CB_DOWNSAMPLING_COEFF_X = 2
        self.CB_DOWNSAMPLING_COEFF_Y = 2
        self.CR_DOWNSAMPLING_COEFF_X = 2
        self.CR_DOWNSAMPLING_COEFF_Y = 2

        self.CATEGORIES_CNT = 12
        self.LENGTHS_CNT = 16
        self.MAX_ZEROS_SEQUENCE_LEN = 15
        self.RUN_SIZE_CNTS = 162

        self.BLOCK_ROWS = 8
        self.BLOCK_COLUMNS = 8

        self.EOB = (0, 0)

        self.category_read_mode = True
        self.run_size_read_mode = True

        self.bits_buffer = ""
        self.current_category = 0
        self.current_block_index = 0

        self.current_block = []

        self.prev = 0

        self.dc_luminance_table_backward = {}
        self.dc_chrominance_table_backward = {}
        self.ac_luminance_table_backward = {}
        self.ac_chrominance_table_backward = {}

    def load_dc_table(self, file_reader):
        dc_categories = file_reader.read(self.CATEGORIES_CNT)
        dc_lengths = file_reader.read(self.LENGTHS_CNT)
        return HuffmanTablesManager.recounstruct_huffman_dc_tree(dc_categories, dc_lengths)

    def load_ac_table(self, file_reader):
        dc_categories = file_reader.read(self.RUN_SIZE_CNTS)
        dc_lengths = file_reader.read(self.LENGTHS_CNT)
        return HuffmanTablesManager.recounstruct_huffman_ac_tree(dc_categories, dc_lengths)

    @staticmethod
    def make_backward_table(forward_table):
        backward_table = {}
        for key in forward_table.keys():
            backward_table[forward_table[key]] = key

        return backward_table

    def load_huffman_tables(self, file_reader):
        self.dc_luminance_table_backward = self.make_backward_table(self.load_dc_table(file_reader))
        self.dc_chrominance_table_backward = self.make_backward_table(self.load_dc_table(file_reader))
        self.ac_luminance_table_backward = self.make_backward_table(self.load_ac_table(file_reader))
        self.ac_chrominance_table_backward = self.make_backward_table(self.load_ac_table(file_reader))

    def ac_decompression_handle_bit(self, bit: str, ac_list, huffman_ac_table):
        self.bits_buffer += bit

        if self.run_size_read_mode:
            if self.bits_buffer in huffman_ac_table:
                run_size = huffman_ac_table[self.bits_buffer]

                if run_size == self.EOB:
                    while len(self.current_block) < 63:
                        self.current_block.append(0)
                    ac_list.append(self.current_block[:])
                    self.current_block_index += 1
                    self.current_block.clear()
                else:
                    for i in range(run_size[0]):
                        self.current_block.append(0)
                    if run_size[1]:
                        self.current_category = run_size[1]
                        self.run_size_read_mode = False
                    else:
                        self.current_block.append(run_size[1])

                self.bits_buffer = ""
        else:
            if len(self.bits_buffer) == self.current_category:
                ac = self.get_dc(self.bits_buffer)
                self.current_block.append(ac)

                self.bits_buffer = ""
                self.run_size_read_mode = True

        if len(self.current_block) >= 63:
            ac_list.append(self.current_block[:])
            self.current_block_index += 1
            self.current_block.clear()
            self.bits_buffer = ""
            self.run_size_read_mode = True

    def ac_decompression(self, file_reader, blocks_cnt, huffman_table):
        ac_list = []
        self.run_size_read_mode = True

        byte_cnt = 0

        while self.current_block_index < blocks_cnt:
            byte = file_reader.read(1)
            byte_cnt += 1

            if not byte:
                break

            for bit in bin(byte[0])[2::].zfill(8):
                if self.current_block_index < blocks_cnt:
                    self.ac_decompression_handle_bit(bit, ac_list, huffman_table)
                else:
                    break

        self.run_size_read_mode = True
        self.bits_buffer = ""
        self.current_block_index = 0
        self.current_block = []
        self.current_category = 0

        return ac_list

    @staticmethod
    def get_dc(num_code):
        if num_code[0] == '0':  # это отрицательное число
            num = -int(inverse_bits(num_code), 2)
            return num

        return int(num_code, 2)

    def dc_decompression_handle_bit(self, bit: str, dc_list, huffman_dc_table):
        self.bits_buffer += bit

        if self.category_read_mode:
            if self.bits_buffer in huffman_dc_table:
                self.current_category = huffman_dc_table[self.bits_buffer]
                if self.current_category == 0:
                    dc_list.append(self.prev)
                    self.current_block_index += 1
                else:
                    self.category_read_mode = False
                self.bits_buffer = ""
        else:
            if len(self.bits_buffer) == self.current_category:
                dc = self.get_dc(self.bits_buffer)

                dc_list.append(dc + self.prev)
                self.prev += dc

                self.bits_buffer = ""

                self.current_block_index += 1
                self.category_read_mode = True

    def dc_decompression(self, file_reader, blocks_cnt, huffman_table):
        dc_list = []
        while self.current_block_index < blocks_cnt:
            byte = file_reader.read(1)

            if not byte:
                break

            for bit in bin(byte[0])[2::].zfill(8):
                if self.current_block_index < blocks_cnt:
                    self.dc_decompression_handle_bit(bit, dc_list, huffman_table)
                else:
                    break

        self.category_read_mode = True
        self.bits_buffer = ""
        self.current_block_index = 0
        self.current_category = 0
        self.prev = 0
        return dc_list

    @staticmethod
    def zigzag_reconversion(coefficients_list: list[int]):
        rows, columns = 8, 8
        matrix = np.zeros((rows, columns))

        i = 0

        x, y = 0, 0

        while True:
            matrix[x, y] = coefficients_list[i]
            i += 1

            if x == columns - 1 and y == rows - 1:
                break

            if (x + y) % 2 == 0:  # Движение вправо-вверх
                if y == 0:
                    x += 1
                elif x == columns - 1:
                    y += 1
                else:
                    x += 1
                    y -= 1
            else:
                if y == rows - 1:
                    x += 1
                elif x == 0:  # Движение влево-вниз
                    y += 1
                else:
                    x -= 1
                    y += 1

        return matrix

    def process_blocks(self, quantized_matricies, quantizationMatrixType):
        inverse_quantized_matricies = []

        for matrix in quantized_matricies:
            dequantized_matrix = self.quantization.dequantization(matrix, quantizationMatrixType)
            idct_matrix = self.dct.inverse_dct(dequantized_matrix)
            inverse_quantized_matricies.append(idct_matrix)

        return inverse_quantized_matricies

    @staticmethod
    def reconstruct_component_to_image_matrix(componet_matricies, size):
        height, width = size

        height -= height % 8
        width -= width % 8

        component = np.zeros((height, width))

        i = 0
        for y in range(0, height, 8):
            for x in range(0, width, 8):
                component[y:y + 8, x:x + 8] = componet_matricies[i]
                i += 1

        return component

    def reconstruct_component_from_lists(self, dc_list, ac_list):
        i = 0
        componet_matricies = []
        for coefficients_list in ac_list:
            coefficients_list.insert(0, dc_list[i])
            componet_matricies.append(self.zigzag_reconversion(coefficients_list))
            i += 1

        return componet_matricies

    def decompression(self, file_path):
        with open(file_path, "rb") as file_reader:
            width = int.from_bytes(file_reader.read(self.BYTES_CNT_TO_WRITE_RESOLUTION), byteorder="big")
            height = int.from_bytes(file_reader.read(self.BYTES_CNT_TO_WRITE_RESOLUTION), byteorder="big")
            resolution = (width, height)

            cb_downsampling_coeff = int.from_bytes(file_reader.read(self.BYTES_CNT_TO_WRITE_DOWNSAMPLE_COEF),
                                                   byteorder="big")
            cr_downsampling_coeff = int.from_bytes(file_reader.read(self.BYTES_CNT_TO_WRITE_DOWNSAMPLE_COEF),
                                                   byteorder="big")

            quality, scaled_luminance_matrix, scaled_chrominance_matrix = QuantizationTableIO.read_quality_and_quantization_matrices(
                file_reader, (self.BLOCK_ROWS, self.BLOCK_COLUMNS))
            self.quantization.set_luminance_matrix(scaled_luminance_matrix)
            self.quantization.set_chrominance_matrix(scaled_chrominance_matrix)
            self.quantization.scale_quantization_matrix(quality)

            self.load_huffman_tables(file_reader)

            blocks_cnt = (width // 8) * (height // 8)

            y_dc_list = self.dc_decompression(file_reader, blocks_cnt, self.dc_luminance_table_backward)
            y_ac_list = self.ac_decompression(file_reader, blocks_cnt, self.ac_luminance_table_backward)
            Y_matricies = self.reconstruct_component_from_lists(y_dc_list, y_ac_list)
            Y_matricies = self.process_blocks(Y_matricies, QuantizationMatrixType.L)
            Y = self.reconstruct_component_to_image_matrix(Y_matricies, resolution)

            cb_dc_list = self.dc_decompression(file_reader, blocks_cnt // (cb_downsampling_coeff ** 2),
                                               self.dc_chrominance_table_backward)
            cb_ac_list = self.ac_decompression(file_reader, blocks_cnt // (cb_downsampling_coeff ** 2),
                                               self.ac_chrominance_table_backward)
            Cb_matricies = self.reconstruct_component_from_lists(cb_dc_list, cb_ac_list)
            Cb_matricies = self.process_blocks(Cb_matricies, QuantizationMatrixType.C)
            Cb = self.reconstruct_component_to_image_matrix(Cb_matricies, (
                resolution[0] // cb_downsampling_coeff, resolution[1] // cb_downsampling_coeff))

            dc_list = self.dc_decompression(file_reader, blocks_cnt // (cr_downsampling_coeff ** 2),
                                            self.dc_chrominance_table_backward)
            ac_list = self.ac_decompression(file_reader, blocks_cnt // (cr_downsampling_coeff ** 2),
                                            self.ac_chrominance_table_backward)
            Cr_matricies = self.reconstruct_component_from_lists(dc_list, ac_list)
            Cr_matricies = self.process_blocks(Cr_matricies, QuantizationMatrixType.C)
            Cr = self.reconstruct_component_to_image_matrix(Cr_matricies, (
                resolution[0] // cr_downsampling_coeff, resolution[1] // cr_downsampling_coeff))

            Cb = self.resample.upSample(Cb, resolution)
            Cr = self.resample.upSample(Cr, resolution)

            new_image_YCbCr = np.dstack((Y, Cb, Cr))
            new_image_RGB = self.color_converter.convert(new_image_YCbCr, ColorSchemes.RGB)

            new_image = Image.fromarray(new_image_RGB)
            new_image.show()

            new_image.save(f"Lenna{self.QUALITY}.png")


if __name__ == "__main__":
    # compressed_image = "images/Lenna_jpeg.txt"
    compressed_image = "images/Lenna512_jpeg.txt"
    # compressed_image = "images/small_image_50x50_jpeg.txt"

    jpeg_decompression = JPEG_Decompression(50)
    jpeg_decompression.decompression(compressed_image)
