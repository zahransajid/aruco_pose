import os
import cv2
import numpy as np
from detect import Detector
import json


class Solver:
    def __init__(self, side_length: float = 14.44) -> None:
        self.LENGTH = side_length / 2
        self.matrices_created = False
        self.focal_length = None
        self.detector = Detector()
        self.object_points = np.array(
            (
                (-self.LENGTH, self.LENGTH, 0),
                (self.LENGTH, self.LENGTH, 0),
                (self.LENGTH, -self.LENGTH, 0),
                (-self.LENGTH, -self.LENGTH, 0),
            ),
            dtype=np.float32,
        )

    def run(self, image: cv2.Mat) -> dict or None:
        """Solves a planar track of an image containing 4 aruco trackers arranged in a square.
        The tracker ids must be 0, 1, 2, 3 in the order top-left, top-right, bottom-left, bottom-right.

        Args:
            image (cv2.Mat): The input image

        Returns:
            dict: Contains keys corners, rotvec, transvec, angles, cam_pos, cam_mat, dist_mat.
            corners - the 4 corners of the square of aruco markers.
            rotvec, transvec - the image space rotation and transformation matrices for the planar track.
            angles - euler angles in radians of the rotation of the plane
            cam_pos - the camera position in world space coordinates.
            cam_mat, dist_mat - the camera and distortion matrices for the input image
        """
        img_h, img_w, img_c = image.shape
        self.focal_length = 1 * img_w
        self.cam_matrix = np.array(
            [
                [self.focal_length, 0, img_h / 2],
                [0, self.focal_length, img_w / 2],
                [0, 0, 1],
            ]
        )
        self.dist_matrix = np.zeros((4, 1), dtype=np.float64)
        results = self.detector.detect_corners(image)
        if results:
            results = np.array(results, dtype=np.float64)
            success, rot_vec, trans_vec = cv2.solvePnP(
                self.object_points,
                results,
                self.cam_matrix,
                self.dist_matrix,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            rmat, jac = cv2.Rodrigues(rot_vec)
            cameraPosition = -np.matrix(rmat).T * np.matrix(trans_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            output = {
                "corners": results,
                "rotvec": rot_vec,
                "transvec": trans_vec,
                "angles": angles,
                "cam_pos": cameraPosition,
                "cam_mat": self.cam_matrix,
                "dist_mat": self.dist_matrix,
            }
            return output
        else:
            return None
