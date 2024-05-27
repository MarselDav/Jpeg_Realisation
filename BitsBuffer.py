class BitsBuffer:
    def __init__(self):
        self.bytes = bytearray()
        self.current_bin_code = ""
        self.file_writer = None

    def set_file_writer(self, file_writer):
        self.file_writer = file_writer

    def add(self, code):
        self.current_bin_code += code
        if len(self.current_bin_code) >= 8:
            self.append()

    def append(self):
        self.bytes.append(int(self.current_bin_code[:8:], 2))
        self.current_bin_code = self.current_bin_code[8::]

    def write(self):
        if self.file_writer is not None:
            if len(self.current_bin_code) > 0:
                self.current_bin_code = self.current_bin_code.ljust(8, "0")
                self.append()
            self.file_writer.write(self.bytes)
            self.bytes.clear()
        else:
            print("file_writer не существует")