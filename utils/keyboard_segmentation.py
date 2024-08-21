import cv2
import numpy as np

from typing import Iterable, List, Tuple, Any

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
Image = np.ndarray

class KeyboardSegmentation:

    def __init__(self, width=1600, height=400) -> None:
        self.homography_width = width
        self.homography_height = height
        self.red_coordinates: List[Point] = []
        self.green_coordinates: List[Point] = []
        self.keyboard_image: Image = None
        self.homography_matrix: Image = None

    def segment_keyboard(self, cam_image: Image, debug=True):
        red_mask = self._get_red_mask(cam_image)
        red_objects = self._detect_red_objects(red_mask)
        if debug:
            for i, (x, y, w, h) in enumerate(red_objects):
                cv2.rectangle(cam_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        try:
            self._calculate_corners_from_red_objects(red_objects)
        except ValueError:
            return
        self._compute_homography()
        self._project_keyboard_image(cam_image)

    @staticmethod
    def _get_red_mask(image: Image) -> Image:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color mask with robust thresholds
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = mask1 + mask2

        return red_mask

    @staticmethod
    def _detect_red_objects(mask: Image) -> List[Rect]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_objects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 10 < w < 30 and 10 < h < 30:  # Filtering based on expected small size
                red_objects.append((x, y, w, h))
        red_objects = sorted(
            red_objects, key=lambda x: (x[1], x[0])
        )  # Sort by y, then by x
        return red_objects

    def _calculate_corners_from_red_objects(self, red_objects: List[Rect]) -> None:
        if len(red_objects) < 3:
            raise ValueError(
                "At least 3 red rectangles are needed to infer the fourth corner."
            )

        red_objects = sorted(
            red_objects, key=lambda x: (x[1], x[0])
        )  # Sort by y, then by x

        top_left = red_objects[0]
        top_right = red_objects[1]
        bottom_left = red_objects[-1]
        # TODO add here some algorithm for choosing the 4 most probable points
        if len(red_objects) == 4:
            bottom_right = red_objects[2]
        else:
            # Infer the missing corner
            top_left_center = (
                top_left[0] + top_left[2] // 2,
                top_left[1] + top_left[3] // 2,
            )
            top_right_center = (
                top_right[0] + top_right[2] // 2,
                top_right[1] + top_right[3] // 2,
            )
            bottom_left_center = (
                bottom_left[0] + bottom_left[2] // 2,
                bottom_left[1] + bottom_left[3] // 2,
            )

            bottom_right_center = (
                bottom_left_center[0] + (top_right_center[0] - top_left_center[0]),
                bottom_left_center[1] + (top_right_center[1] - top_left_center[1]),
            )
            bottom_right = (
                bottom_right_center[0] - top_right[2] // 2,
                bottom_right_center[1] - top_right[3] // 2,
                top_right[2],
                top_right[3],
            )

        corners = [
            (top_left[0], top_left[1]),
            (top_right[0] + top_right[2], top_right[1]),
            (bottom_left[0], bottom_left[1] + bottom_left[3]),
            (bottom_right[0] + bottom_right[2], bottom_right[1] + bottom_right[3]),
        ]
        self.red_coordinates = corners

    def _compute_homography(self):
        src_points = np.array(self.red_coordinates, dtype=np.float32)
        dst_points = np.array(
            [
                [0, 0],
                [self.homography_width, 0],
                [0, self.homography_height],
                [self.homography_width, self.homography_height],
            ],
            dtype=np.float32,
        )
        dst_points = np.array(
            [
                [0, 0],
                [self.homography_width, 0],
                [self.homography_width, self.homography_height],
                [0, self.homography_height],
            ],
            dtype=np.float32,
        )
        homography_matrix, _ = cv2.findHomography(src_points, dst_points)
        self.homography_matrix = homography_matrix

    def _project_keyboard_image(self, cam_image):
        homograph_image = cv2.warpPerspective(
            cam_image,
            self.homography_matrix,
            (self.homography_width, self.homography_height),
        )
        homograph_image = cv2.rotate(homograph_image, cv2.ROTATE_180)
        homograph_image = cv2.flip(homograph_image, 1)
        self.keyboard_image = homograph_image

    def project_point(self, point, inverse=False) -> Point:
        homography_matrix = (
            np.linalg.inv(self.homography_matrix) if inverse else self.homography_matrix
        )
        src_point = np.array([[point]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, homography_matrix)
        return tuple(dst_point[0][0])
