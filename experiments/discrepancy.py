import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    B = 10000
    N = 100
    data = np.random.random([B, N]) * np.random.random([B, 1])

    S = np.sum(data, axis=1, keepdims=True)
    data_mean = np.sum(data * data / S, axis=1)
    data_max = np.max(data, axis=1)

    indices = np.argsort(data_max)

    print(np.sum(data_max / data_mean) / B)

    x = np.arange(B)
    plt.plot(x, data_max[indices], 'g')
    plt.plot(x, data_mean[indices], 'b')
    plt.show()

    # As N grows larger, the sequence converge to 3/2
    # (3x - 1) / 2
