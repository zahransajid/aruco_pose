import cv2
import numpy as np
from cv2 import aruco


class Detector:
    def __init__(self, aruco_dict=aruco.DICT_4X4_50) -> None:
        self.params = aruco.DetectorParameters_create()
        self.aruco_dict = aruco.Dictionary_get(aruco_dict)

    def detectFour(self, img: cv2.Mat) -> tuple or None:
        """Detects and returns tuple of id,corner pairs for the included 4 marker sheet.

        Args:
            img (cv2.Mat): Input image

        Returns:
            tuple or None: zipped tuple of id, corner point pairs
        """
        corners, ids, _ = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.params
        )
        ids = ids.flatten()
        if len(ids) != 4:
            return None
        else:
            return zip(ids, corners)

    def detectAll(self, img: cv2.Mat) -> tuple:
        """Returns all aruco markers present in image as id, corner point pairs

        Args:
            img (cv2.Mat): Input image

        Returns:
            tuple: zipped (id, corner points) pairs
        """
        corners, ids, _ = cv2.aruco.detectMarkers(
            img, self.aruco_dict, parameters=self.params
        )
        ids = ids.flatten()
        return zip(ids, corners)

    def detect_corners(self, img: cv2.Mat) -> tuple:
        """Returns the 4 corner points of the included tracker image in the order top-left, top-right, bottom-right, bottom-left.

        Args:
            img (cv2.Mat): Input image

        Returns:
            tuple: Tuple of pixel coordinates of points
        """
        # The order of corners for each aruco marker is top-left,top-right,bottom-right,bottom-left.
        results = self.detectFour(img)
        if not results:
            return None
        results = dict(results)
        for key in results.keys():
            results[key] = results[key].reshape((4, 2))
        topLeft = np.array(results[0][0], dtype=np.int32)
        topRight = np.array(results[1][1], dtype=np.int32)
        bottomLeft = np.array(results[2][3], dtype=np.int32)
        bottomRight = np.array(results[3][2], dtype=np.int32)
        return (topLeft, topRight, bottomRight, bottomLeft)

    def getPose(self, img: cv2.Mat, markerCorners: np.ndarray) -> dict:
        """Returns rotation, translation vectors, camera and distortion matrices in a dictionary
        for a given set of corners of an aruco marker
        Args:
            img (cv2.Mat): Input image used to find camera coefficients
            markerCorners (np.ndarray): corners of the aruco marker

        Returns:
            dict: The keys are
            "rotvec",
            "transvec",
            "marker_points",
            "cam_mat",
            "dist_mat",
        """
        img_h, img_w, img_c = img.shape
        focal_length = 1 * img_w
        cam_matrix = np.array(
            [
                [focal_length, 0, img_h / 2],
                [0, focal_length, img_w / 2],
                [0, 0, 1],
            ]
        )
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(
            markerCorners, 0.02, cam_matrix, dist_matrix
        )
        out = {
            "rotvec": rvec,
            "transvec": tvec,
            "marker_points": markerPoints,
            "cam_mat": cam_matrix,
            "dist_mat": dist_matrix,
        }
        return out
