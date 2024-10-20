import cv2
import numpy as np
import random
import math
import os
from queue import PriorityQueue
from utilities import write_gif


def search_path(minkowsky_map, start, end):

    def to_queue(p, prev, distp):
        if prev is None:
            return (distp, (p[0], p[1], 0))
        else:
            return (prev[2] + distp, (p[0], p[1], prev[2] + distp, prev))

    q = PriorityQueue()
    q.put(to_queue((start[0], start[1]), None, 0.0))

    def off_range(point):
        return p[1] < 0 or p[1] >= minkowsky_map.shape[0] or p[0] < 0 or p[0] >= minkowsky_map.shape[1]

    out = None
    visited = np.zeros((minkowsky_map.shape[0], minkowsky_map.shape[1]), dtype=np.float32)
    while not q.empty():
        p = q.get()[1]
        if p[0] == end[0] and p[1] == end[1]:
            out = p
            break
        if off_range(p) or visited[p[1], p[0]] > 0.5 or minkowsky_map[p[1], p[0]] > 0.5:
            continue
        visited[p[1], p[0]] = 1
        q.put(to_queue((p[0] + 0, p[1] - 1), p, 1))
        q.put(to_queue((p[0] - 1, p[1] + 0), p, 1))
        q.put(to_queue((p[0] + 1, p[1] + 0), p, 1))
        q.put(to_queue((p[0] + 0, p[1] + 1), p, 1))
        q.put(to_queue((p[0] - 1, p[1] - 1), p, math.sqrt(2)))
        q.put(to_queue((p[0] + 1, p[1] - 1), p, math.sqrt(2)))
        q.put(to_queue((p[0] - 1, p[1] + 1), p, math.sqrt(2)))
        q.put(to_queue((p[0] + 1, p[1] + 1), p, math.sqrt(2)))

    path = []
    if out is None:
        return path

    p = out
    while True:
        path.append((p[0], p[1]))
        if len(p) > 3:
            p = p[3]
        else:
            break
    path.reverse()
    return path


def draw_map(size, obj_size, map_complexity):
    canvas = np.zeros((size[0], size[1]), dtype=np.float32)
    minkowsky = np.zeros((size[0], size[1]), dtype=np.float32)

    half_obj_size = obj_size / 2
    for i in range(map_complexity):
        sx = random.randint(2, 15) * size[1] / 100
        sy = random.randint(2, 15) * size[0] / 100
        py = random.randint(half_obj_size, size[0] - half_obj_size)
        px = random.randint(half_obj_size, size[1] - half_obj_size)
        cv2.rectangle(canvas, (int(px - sx), int(py - sy)), (int(px + sx), int(py + sy)), (1.0, 1.0, 1.0), -1)
        cv2.rectangle(minkowsky, (int(px - sx - half_obj_size), int(py - sy - half_obj_size)), (int(px + sx + half_obj_size), int(py + sy + half_obj_size)), (1.0, 1.0, 1.0), -1)

    return canvas, minkowsky


def draw_path(space, obj_size, path, length_modifier=0.2):
    path_length = len(path)
    if length_modifier < 1:
        fixed_length = int(path_length * length_modifier) + 1
    else:
        fixed_length = int(length_modifier)
    canvas = np.zeros((fixed_length, space.shape[0], space.shape[1]), dtype=np.float32)
    half_obj_size = obj_size // 2
    # draw target configuration on the first frame
    cv2.circle(canvas[0], (path[path_length - 1][0], path[path_length - 1][1]), half_obj_size, (1.0, 1.0, 1.0), -1)
    # then draw a series of movement
    step = (path_length - 1) * 1.0 / (fixed_length - 2)
    for i in range(1, fixed_length):
        canvas[i] = space
        p = path[int((i - 1) * step)]
        cv2.circle(canvas[i], (int(p[0]), int(p[1])), half_obj_size, (1.0, 1.0, 1.0), -1)
    return canvas


def get_valid_data(map_size, obj_size, map_complexity, length_modifier):
    while True:
        space, minkowsky = draw_map(map_size, obj_size, map_complexity)
        half_obj_size = obj_size // 2
        map_offset_x = map_size[1] - half_obj_size
        map_offset_y = map_size[0] - half_obj_size
        path = search_path(minkowsky, (half_obj_size, random.randint(half_obj_size, map_offset_y)), (map_offset_x, random.randint(half_obj_size, map_offset_y)))
        if len(path) <= 0:
            print("no path is possible!")
        else:
            frames = draw_path(space, obj_size, path, length_modifier)
            break
    return frames


def to_numpy(frame_list, map_size):
    frames = np.zeros((len(frame_list), map_size[0], map_size[1]), dtype=np.float32)
    for i in range(len(frame_list)):
        frames[i, :] = np.reshape(frame_list[i], map_size)
    return frames


def plot_path(frames, map_size):
    temp = np.copy(frames)
    temp[:, 0, :] = 1
    temp[:, :, 0] = 1
    cv2.imshow("a", np.reshape(np.transpose(temp, (1, 0, 2)), (map_size[0], -1)))
    cv2.moveWindow("a", 100, 100)
    cv2.waitKey(-1)


def toGif(frames, filename):
    imgs = []
    for i in range(frames.shape[0]):
        img = cv2.cvtColor((frames[i] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        imgs.append(img)
    write_gif(imgs, filename, fps=5)


if __name__ == "__main__":
    frames = get_valid_data((100, 100), 10, 6, 0.2)
    artifact_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")
    sample_path = os.path.join(artifact_path, "sample_path.gif")
    toGif(frames, sample_path)
    plot_path(frames, (100, 100))
