import cv2
import numpy as np
from cv2 import aruco

OUT_DIR = "markers/"


def main(len_markers: int):
    ARUCO_DICT = aruco.DICT_4X4_50
    ARUCO_SIDE = 512
    tag = np.zeros((ARUCO_SIDE, ARUCO_SIDE, 1), dtype="uint8")
    aruco_dict = aruco.Dictionary_get(ARUCO_DICT)
    markers = [
        cv2.aruco.drawMarker(aruco_dict, i, ARUCO_SIDE, borderBits=1)
        for i in range(len_markers)
    ]
    for i, tag in enumerate(markers):
        cv2.imwrite(f"{OUT_DIR}{i}.png", tag)


if __name__ == "__main__":
    main(4)
