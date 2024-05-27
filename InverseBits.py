def inverse_bits(bin_num):
    invert = {"0": "1", "1": "0"}
    bit_length = len(bin_num)

    inverse_num = ""
    for i in range(bit_length):
        inverse_num += invert[bin_num[i]]

    return inverse_num.zfill(bit_length)