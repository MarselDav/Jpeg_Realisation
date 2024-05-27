import numpy as np


class DiscreteCosinTransformation:
    def __init__(self):
        self.N = 8
        self.C1 = np.sqrt(1 / self.N)
        self.C2 = np.sqrt(2 / self.N)
        self.H = self.getH()
        self.H_T = self.H.transpose()

    def getH(self) -> np.array:
        H = np.zeros((self.N, self.N))
        for j in range(self.N):
            H[0, j] = self.C1

        for i in range(1, self.N):
            for j in range(self.N):
                H[i, j] = self.C2 * np.cos(((2 * j + 1) * i * np.pi) / (2 * self.N))

        return H

    def forward_dct(self, A):
        return np.matmul(np.matmul(self.H, A), self.H_T)

    def inverse_dct(self, C):
        return np.matmul(np.matmul(self.H_T, C), self.H)


if __name__ == "__main__":
    dct = DiscreteCosinTransformation()

    import time

    cnt = 10000
    result = 0
    for i in range(cnt):
        start = time.time_ns()

        np.random.seed(i)
        A = np.random.randint(0, 255, (8, 8))
        A_shift = A - (2 ** 7 - 1)

        # fdct = dct.forward_dct(A_shift)
        fdct = dct.forward_dct(A_shift)
        # idct = dct.inverse_dct(fdct)
        # A_new = idct + (2 ** 7 - 1)
        # A_new = np.round(A_new)
        # A_new = np.ndarray.astype(A_new, dtype=np.int32)

        end = time.time_ns()

        result += end - start
    print(result // cnt)
