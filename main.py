from detect import Detector
from camera_pose import Solver
import cv2
import numpy as np
import os
from copy import deepcopy


def test_detect_4markers(test_img: str):
    dct = Detector()
    img = cv2.imread(test_img)
    results = dct.detect_corners(img)
    for r in results:
        cv2.circle(img, r, 50, (0, 0, 255), cv2.FILLED)
    cv2.line(img, results[0], results[1], (0, 255, 0), 10)
    cv2.line(img, results[1], results[2], (0, 255, 0), 10)
    cv2.line(img, results[2], results[3], (0, 255, 0), 10)
    cv2.line(img, results[3], results[0], (0, 255, 0), 10)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def test_detect_4markers2(test_img: str):
    dct = Detector()
    img = cv2.imread(test_img)
    results = dct.detectAll(img)
    results_copy = deepcopy(results)
    poses = [dct.getPose(img, result[1]) for result in results]
    for i, (markerID, markerCorner) in enumerate(results_copy):
        pose = poses[i]
        corners = markerCorner.reshape((4, 2))
        (topLeft, topRight, bottomRight, bottomLeft) = corners
        # convert each of the (x, y)-coordinate pairs to integers
        upVector = np.array([[0, 0, 0], [0, 0, 0.01]], dtype=np.float32)
        leftVector = np.array([[0, 0, 0], [0, 0.01, 0]], dtype=np.float32)
        rightVector = np.array([[0, 0, 0], [0.01, 0, 0]], dtype=np.float32)
        rot_vec = pose["rotvec"]
        trans_vec = pose["transvec"]
        cam_matrix = pose["cam_mat"]
        dist_matrix = pose["dist_mat"]
        projectionUp, _ = cv2.projectPoints(
            upVector, rot_vec, trans_vec, cam_matrix, dist_matrix
        )
        projectionLeft, _ = cv2.projectPoints(
            leftVector, rot_vec, trans_vec, cam_matrix, dist_matrix
        )
        projectionRight, _ = cv2.projectPoints(
            rightVector, rot_vec, trans_vec, cam_matrix, dist_matrix
        )
        projectionUp = projectionUp.flatten()
        projectionLeft = projectionLeft.flatten()
        projectionRight = projectionRight.flatten()
        p1 = np.array([projectionUp[0], projectionUp[1]], dtype=np.int32)
        p2 = np.array([projectionUp[2], projectionUp[3]], dtype=np.int32)
        p3 = np.array([projectionLeft[0], projectionLeft[1]], dtype=np.int32)
        p4 = np.array([projectionLeft[2], projectionLeft[3]], dtype=np.int32)
        p5 = np.array([projectionRight[0], projectionRight[1]], dtype=np.int32)
        p6 = np.array([projectionRight[2], projectionRight[3]], dtype=np.int32)
        topRight = (int(topRight[0]), int(topRight[1]))
        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
        topLeft = (int(topLeft[0]), int(topLeft[1]))
        cv2.line(img, topLeft, topRight, (0, 255, 0), 10)
        cv2.line(img, topRight, bottomRight, (0, 255, 0), 10)
        cv2.line(img, bottomRight, bottomLeft, (0, 255, 0), 10)
        cv2.line(img, bottomLeft, topLeft, (0, 255, 0), 10)
        cv2.arrowedLine(img, p5, p6, (0, 0, 255), 17)
        cv2.arrowedLine(img, p3, p4, (255, 255, 0), 17)
        cv2.arrowedLine(img, p1, p2, (255, 0, 0), 17)
        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
        cv2.circle(img, (cX, cY), 4, (0, 0, 255), -1)
        cv2.putText(
            img,
            str(markerID),
            (topLeft[0], topLeft[1] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 255),
            7,
        )

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def test_pose_4markers(test_images_dir: str, interval: int = 10):
    out = {}
    est = Solver()
    imgs = os.listdir(test_images_dir)
    print(len(imgs))
    imgs = imgs[::interval]
    for no, img in enumerate(imgs):
        img = cv2.imread(os.path.join("frames/", img))
        results = est.run(img)
        print(f"processing image {no}")
        if results:
            upVector = np.array([[0, 0, 0], [0, 0, 3]], dtype=np.float32)
            leftVector = np.array([[0, 0, 0], [0, 3, 0]], dtype=np.float32)
            rightVector = np.array([[0, 0, 0], [3, 0, 0]], dtype=np.float32)
            rot_vec = results["rotvec"]
            trans_vec = results["transvec"]
            cam_matrix = results["cam_mat"]
            dist_matrix = results["dist_mat"]
            out[no] = {
                "camera_position": list(results["cam_pos"].tolist()),
                "rotation_vectors": list(rot_vec.tolist()),
                "angles": list(results["angles"]),
            }
            projectionUp, _ = cv2.projectPoints(
                upVector, rot_vec, trans_vec, cam_matrix, dist_matrix
            )
            projectionLeft, _ = cv2.projectPoints(
                leftVector, rot_vec, trans_vec, cam_matrix, dist_matrix
            )
            projectionRight, _ = cv2.projectPoints(
                rightVector, rot_vec, trans_vec, cam_matrix, dist_matrix
            )
            corners = np.array(results["corners"], dtype=np.int32)
            projectionUp = projectionUp.flatten()
            projectionLeft = projectionLeft.flatten()
            projectionRight = projectionRight.flatten()
            p1 = np.array([projectionUp[0], projectionUp[1]], dtype=np.int32)
            p2 = np.array([projectionUp[2], projectionUp[3]], dtype=np.int32)
            p3 = np.array([projectionLeft[0], projectionLeft[1]], dtype=np.int32)
            p4 = np.array([projectionLeft[2], projectionLeft[3]], dtype=np.int32)
            p5 = np.array([projectionRight[0], projectionRight[1]], dtype=np.int32)
            p6 = np.array([projectionRight[2], projectionRight[3]], dtype=np.int32)
            cv2.line(img, corners[0], corners[1], (0, 255, 0), 10)
            cv2.line(img, corners[1], corners[2], (0, 255, 0), 10)
            cv2.line(img, corners[2], corners[3], (0, 255, 0), 10)
            cv2.line(img, corners[3], corners[0], (0, 255, 0), 10)
            cv2.line(img, corners[0], corners[2], (0, 255, 0), 10)
            cv2.line(img, corners[1], corners[3], (0, 255, 0), 10)
            cv2.arrowedLine(img, p5, p6, (0, 0, 255), 17)
            cv2.arrowedLine(img, p3, p4, (255, 255, 0), 17)
            cv2.arrowedLine(img, p1, p2, (255, 0, 0), 17)
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", img)
            cv2.waitKey(0)
    # with open("frames.json", "w") as f:
    #     json.dump(out, f)
    #     f.close()


if __name__ == "__main__":
    test_detect_4markers2(r"frames_limited\0247.jpg")
    test_detect_4markers(r"frames_limited\0247.jpg")
    test_pose_4markers(r"frames_limited/", interval=2)
