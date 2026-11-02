"""Shared helpers: the on-disk frame store (concatenated JPEGs + offset index)."""
import json
import os

import cv2
import numpy as np


class FrameWriter:
    """Append JPEG frames to one file; random access later via an offset index.

    Chosen over mp4 so that any frame can be fetched by number in O(1) with no
    keyframe seeking, which is what a step-by-step review page needs.
    """

    def __init__(self, directory, quality=85):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.f = open(os.path.join(directory, "frames.bin"), "wb")
        self.offsets = []
        self.q = quality

    def write(self, img_bgr):
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        assert ok
        self.offsets.append(self.f.tell())
        self.f.write(buf.tobytes())

    def close(self):
        self.offsets.append(self.f.tell())
        self.f.close()
        np.save(os.path.join(self.dir, "frames_idx.npy"), np.array(self.offsets, dtype=np.int64))


class FrameStore:
    def __init__(self, directory):
        self.dir = directory
        self.idx = np.load(os.path.join(directory, "frames_idx.npy"))
        self.mm = np.memmap(os.path.join(directory, "frames.bin"), dtype=np.uint8, mode="r")

    def __len__(self):
        return len(self.idx) - 1

    def raw(self, i):
        return self.mm[self.idx[i]:self.idx[i + 1]].tobytes()

    def get(self, i):
        return cv2.imdecode(np.frombuffer(self.mm[self.idx[i]:self.idx[i + 1]], dtype=np.uint8), cv2.IMREAD_COLOR)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)
