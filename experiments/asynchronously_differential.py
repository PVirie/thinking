import cv2
import random


src = cv2.imread("assets/1.png")
dst = cv2.resize(cv2.imread("assets/2.jpeg"), (src.shape[1], src.shape[0]))


def next_order(total):
    buff = list(range(total))
    random.shuffle(buff)
    for i in buff:
        yield i


if __name__ == '__main__':

    steps = 100
    total = src.shape[0] * src.shape[1]
    count_per_step = total // steps

    gen = next_order(total)

    for i in range(steps):
        for j in range(count_per_step):
            index = next(gen)
            src[index // src.shape[1], index % src.shape[1], :] = dst[index // src.shape[1], index % src.shape[1], :]
        cv2.imshow("disp", src)
        cv2.waitKey(1)
