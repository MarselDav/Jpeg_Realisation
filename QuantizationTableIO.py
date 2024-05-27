from Quantization import Quantization
import numpy as np


class QuantizationTableIO:

    @staticmethod
    def write_quantization_matrix(file_writer, maxtrix):
        quantization_bytearray = bytearray()
        for y in range(maxtrix.shape[0]):
            for x in range(maxtrix.shape[1]):
                quantization_bytearray.append(maxtrix[y][x])

        file_writer.write(quantization_bytearray)

    @staticmethod
    def write_quality_and_quantization_matrices(file_writer, quality, quantization_class : Quantization):
        file_writer.write(quality.to_bytes(1, byteorder="big"))
        QuantizationTableIO.write_quantization_matrix(file_writer, quantization_class.scaled_luminance_matrix)
        QuantizationTableIO.write_quantization_matrix(file_writer, quantization_class.scaled_chrominance_matrix)


    @staticmethod
    def read_quantization_matrix(file_reader, shape):
        maxtrix = np.zeros(shape)
        for y in range(shape[0]):
            for x in range(shape[1]):
                maxtrix[y][x] = int.from_bytes(file_reader.read(1), byteorder="big")

        return maxtrix

    @staticmethod
    def read_quality_and_quantization_matrices(file_reader, shape):
        quality = int.from_bytes(file_reader.read(1), byteorder="big")
        scaled_luminance_matrix = QuantizationTableIO.read_quantization_matrix(file_reader, shape)
        scaled_chrominance_matrix = QuantizationTableIO.read_quantization_matrix(file_reader, shape)

        return quality, scaled_luminance_matrix, scaled_chrominance_matrix