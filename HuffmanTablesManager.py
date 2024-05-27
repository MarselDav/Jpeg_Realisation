class HuffmanTablesManager:
    @staticmethod
    def write_dc_table(file_writer, dc_huffman_table):
        length_bytearray = bytearray(16)

        dc_huffman_list = [(len(dc_huffman_table[i]), i) for i in range(len(dc_huffman_table))]
        dc_huffman_list.sort(key=lambda x: x[0])

        length_list = [element[0] for element in dc_huffman_list]
        categories_bytearray = bytearray(element[1] for element in dc_huffman_list)

        for i in range(1, len(length_bytearray) + 1):
            length_bytearray[i - 1] = length_list.count(i)

        file_writer.write(categories_bytearray)
        file_writer.write(length_bytearray)

    @staticmethod
    def recounstruct_huffman_dc_tree(categories_bytearray, length_bytearray):
        huffman_table = {}
        current_code = 0
        category_idx = 0

        # Генерация кодов Хаффмана
        for length in range(1, len(length_bytearray) + 1):
            for _ in range(length_bytearray[length - 1]):
                code = bin(current_code)[2:].zfill(length)
                huffman_table[categories_bytearray[category_idx]] = code
                current_code += 1
                category_idx += 1
            current_code <<= 1

        return huffman_table

    @staticmethod
    def run_size_to_num(run, size):
        num = 0
        num = num | run
        num = num << 4
        num = num | size

        return num

    @staticmethod
    def num_to_run_size(num):
        size = num & 15
        num = num >> 4
        run = num & 15

        return run, size

    @staticmethod
    def write_ac_table(file_writer, ac_huffman_table):
        length_bytearray = bytearray(16)

        ac_huffman_list = [(len(ac_huffman_table[key]), HuffmanTablesManager.run_size_to_num(key[0], key[1])) for key in
                           ac_huffman_table]
        ac_huffman_list.sort(key=lambda x: x[0])

        length_list = [element[0] for element in ac_huffman_list]
        categories_bytearray = bytearray(element[1] for element in ac_huffman_list)

        for i in range(1, len(length_bytearray) + 1):
            length_bytearray[i - 1] = length_list.count(i)

        file_writer.write(categories_bytearray)
        file_writer.write(length_bytearray)

    @staticmethod
    def recounstruct_huffman_ac_tree(categories_bytearray, length_bytearray):
        huffman_table = {}
        current_code = 0
        category_idx = 0

        # Генерация кодов Хаффмана
        for length in range(1, len(length_bytearray) + 1):
            for _ in range(length_bytearray[length - 1]):
                code = bin(current_code)[2:].zfill(length)
                huffman_table[HuffmanTablesManager.num_to_run_size(categories_bytearray[category_idx])] = code
                current_code += 1
                category_idx += 1
            current_code <<= 1

        return huffman_table
