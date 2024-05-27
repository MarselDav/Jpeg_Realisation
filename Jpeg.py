from JpegCompression import JPEG_Compression
from JpegDecompression import JPEG_Decompression

class JPEG:
    def __init__(self, quality):
        self.jpeg_compression = JPEG_Compression(quality)
        self.jpeg_decompression = JPEG_Decompression(quality)


    def compression(self, file_path):
        self.jpeg_compression.compression(file_path)

    def decompression(self, file_path):
        self.jpeg_decompression.decompression(file_path)


if __name__ == "__main__":
    original_image = "images/Lenna512.png"
    # original_image = "images/single_color_image.png"
    # original_image = "images/random_color_image.png"

    compressed_image = "images/Lenna512_jpeg.txt"

    # compressed_image = "images/single_color_image_jpeg.txt"
    # compressed_image = "images/random_color_image_jpeg.txt"

    import os

    original_size = 786_432
    # with open(original_image, "rb") as file:
    #     file.seek(0, os.SEEK_END)
    #     original_size = file.tell()
    #     print(original_size)
    #
    # x = 4096 + 42088 + 4096 + 11324 + 4096 + 11526
    # print(x)

    jpeg = JPEG(100)
    jpeg.compression(original_image)
    jpeg.decompression(compressed_image)
    # with open(compressed_image, "rb") as file:
    #     file.seek(0, os.SEEK_END)
    #     print(file.tell())

    # coeffs = []
    # quality = range(1, 101)
    # for q in quality:
    #     jpeg = JPEG(q)
    #     jpeg.compression(original_image)
    #
    #     with open(compressed_image, "rb") as file:
    #         file.seek(0, os.SEEK_END)
    #         coeffs.append(original_size // file.tell())
    #
    # print(coeffs)