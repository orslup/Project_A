import cv2
import numpy as np
from itertools import combinations

from typing import Iterable, List, Tuple, Any

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
Image = np.ndarray


class KeyboardSegmentation:
    COOLDOWN_FRAMES = 20 * 4  # approximatly 4 seconds
    def __init__(self, width=1600, height=400) -> None:
        self.homography_width = width
        self.homography_height = height
        self.red_coordinates: List[Point] = []
        self.green_coordinates: List[Point] = []
        self.keyboard_image: Image = None
        self.homography_matrix: Image = None
        self.cooldown_frames = self.COOLDOWN_FRAMES

    def segment_keyboard(self, cam_image: Image, debug=True):
        src_points, dst_points = self.get_matching_points(cam_image, how='red')
        if src_points is None or dst_points is None:
            return
        if debug:
            try:
                src_points_int = src_points.astype(np.uint8)
                for point in src_points_int:
                    x, y = point
                    cam_image = cv2.circle(cam_image, (y, x), radius=10, color=(0, 0, 255), thickness=2)
            except Exception as e:
                print(str(e))
            # for i, (x, y) in enumerate(src_points):
            #     cv2.rectangle(cam_image, (float(x), float(y)), (float(x) + 20, float(y) + 20), (0, 0, 255), 2)
        if self.keyboard_image is not None and self.cooldown_frames > 0:
            self.cooldown_frames -= 1
            return
        self.cooldown_frames = self.COOLDOWN_FRAMES

        self.homography_matrix = self._compute_homography(src_points, dst_points)
        keyboard_image = self._project_keyboard_image(cam_image, self.homography_matrix)
        if self._is_keyboard_image(keyboard_image):
            self.keyboard_image = keyboard_image

    @staticmethod
    def _get_mask(image: Image, color='red') -> Image:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red color mask with robust thresholds
        if color == 'red':
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            mask = mask1 + mask2
        elif color == 'green':
            lower_green = np.array([45,100,100])
            upper_green = np.array([75,255,255])

            mask = cv2.inRange(hsv, lower_green, upper_green)
        else:
            raise TypeError(f"Color {color} is invalid")

        return mask
    
    @staticmethod
    def _get_rectangle_middle(rect: Rect) -> Point:
        return (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2)

    @staticmethod
    def _detect_objects(mask: Image) -> List[Rect]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 10 < w < 30 and 10 < h < 30:  # Filtering based on expected small size
                objects.append((x, y, w, h))
        objects = sorted(
            objects, key=lambda x: (x[1], x[0])
        )  # Sort by y, then by x
        return objects

    def _calculate_edges_from_green_objects(self, green_objects: List[Rect], require_all_points=True):
        if len(green_objects) < 4:
            if not require_all_points:
                self.green_coordinates = [(green_object[0], green_object[1]) for green_object in green_objects]
            return
        
        green_objects = sorted(
            green_objects, key=lambda x: (x[1], x[0])
        )  # Sort by y, then by x
        top_edge = green_objects[0]
        left_edge = green_objects[1]
        right_edge = green_objects[2]
        bottom_edge = green_objects[3]

        # edges = [
        #     (top_edge[0], top_edge[1] + top_edge[3] // 2),
        #     (left_edge[0], left_edge[1]),
        #     (right_edge[0], right_edge[1]),
        #     (bottom_edge[0], bottom_edge[1])
        # ]
        edges = list(map(self._get_rectangle_middle, (top_edge,
                                                      left_edge,
                                                      right_edge,
                                                      bottom_edge)))

        # edges = np.array([top_edge, left_edge, right_edge, bottom_edge], dtype=np.float32)
        self.green_coordinates = edges

    def _calculate_corners_from_red_objects(self, red_objects: List[Rect], require_all_points=True) -> None:
        if len(red_objects) < 4:
            if not require_all_points:
                self.red_coordinates = np.array(red_objects, dtype=np.float32)
            return

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

        # corners = [
        #     (top_left[0], top_left[1]),
        #     (top_right[0] + top_right[2], top_right[1]),
        #     (bottom_left[0], bottom_left[1] + bottom_left[3]),
        #     (bottom_right[0] + bottom_right[2], bottom_right[1] + bottom_right[3]),
        # ]
        corners = list(map(self._get_rectangle_middle, (top_left,
                                                top_right,
                                                bottom_left,
                                                bottom_right)))

        self.red_coordinates = corners
    
    def _get_homography_corners(self):
        return [[0, 0], # Top Left
                [self.homography_width, 0], # Top Right
                [self.homography_width, self.homography_height], # Bottom Right
                [0, self.homography_height], # Bottom Left
                ]
    
    def _get_homography_edges(self):
        return [[self.homography_width / 2, 0],        # Top center
                [0, self.homography_height / 2],       # Left center
                [self.homography_width, self.homography_height / 2],   # Right center
                [self.homography_width / 2, self.homography_height]    # Bottom center
                ]

    
    def get_matching_points(self, cam_image: Image, how='red') -> Tuple[np.ndarray, np.ndarray]:
        """
        returns list of src and dest points
        :param how: red - take first red, then green
                    green - take first green, then red
                    ransac - take combinations and try to infer
        """
        require_all_points = how != 'ransac'  # in native methods need atleast 4
        red_mask = self._get_mask(cam_image, color='red')
        green_mask = self._get_mask(cam_image, color='green')
        red_objects = self._detect_objects(red_mask)
        green_objects = self._detect_objects(green_mask)
        self._calculate_corners_from_red_objects(red_objects, require_all_points=require_all_points)
        self._calculate_edges_from_green_objects(green_objects, require_all_points=require_all_points)
        
        # run ransac on points:
        if (how == 'ransac'):
            return self._homography_ransac(cam_image)
        # return red points
        if (how == 'red' and len(self.red_coordinates) == 4) or \
            (how == 'green' and len(self.green_coordinates) < 4
                            and len(self.red_coordinates) == 4):
            return np.array(self.red_coordinates, dtype=np.float32), \
                    np.array(self._get_homography_corners(), dtype=np.float32),
        # return green points
        if (how == 'green' and len(self.green_coordinates) == 4) or \
            (how == 'red' and len(self.red_coordinates) < 4
                            and len(self.green_coordinates) == 4):
            return np.array(self.green_coordinates, dtype=np.float32), \
                    np.array(self._get_homography_edges(), dtype=np.float32),
        return None, None

    def _compute_homography(self, src_points, dst_points):
        homography_matrix, _ = cv2.findHomography(src_points, dst_points)
        return homography_matrix

    def _project_keyboard_image(self, cam_image: Image, homography_matrix: np.ndarray) -> Image:
        homograph_image = cv2.warpPerspective(
            cam_image,
            homography_matrix,
            (self.homography_width, self.homography_height),
        )
        homograph_image = cv2.rotate(homograph_image, cv2.ROTATE_180)
        homograph_image = cv2.flip(homograph_image, 1)
        return homograph_image
    
    @staticmethod
    def _calc_white_intensity(image: Image) -> float:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        white_intensity = float(np.sum(gray_image) / (255.0 * gray_image.size))
        return white_intensity

    def _homography_ransac(self, cam_image: Image):
        corners = self._get_homography_corners()
        edges = self._get_homography_edges()
        best_src = None
        best_dst = None
        max_white_intensity = 0
        for num_red in range(1, len(self.red_coordinates) + 1):
            num_green = 4 - num_red
            if len(self.green_coordinates) < num_green:
                continue
            for red_points in combinations(self.red_coordinates, num_red):
                for green_points in combinations(self.green_coordinates, num_green):
                    for corner_points in combinations(corners, num_red):
                        for edges_points in combinations(edges, num_green):
                            print(f"{red_points=} {green_points=}")
                            src_points = np.array(list(red_points) + list(green_points) ,dtype=np.float32)
                            dst_points = np.array(list(corner_points) + list(edges_points) ,dtype=np.float32)
                            homography_matrix = self._compute_homography(src_points, dst_points)
                            keyboard_image = self._project_keyboard_image(cam_image, homography_matrix)
                            white_intensity = self._calc_white_intensity(keyboard_image)
                            if white_intensity > max_white_intensity:
                                max_white_intensity = white_intensity
                                best_src = src_points
                                best_dst = dst_points
        print(f"{max_white_intensity=}")
        return best_src, best_dst

    def project_point(self, point, inverse=False) -> Point:
        homography_matrix = (
            np.linalg.inv(self.homography_matrix) if inverse else self.homography_matrix
        )
        src_point = np.array([[point]], dtype=np.float32)
        dst_point = cv2.perspectiveTransform(src_point, homography_matrix)
        return tuple(dst_point[0][0])

    def _is_keyboard_image(self, keyboard_image: Image) -> bool:
        white_intensity = self._calc_white_intensity(keyboard_image)
        return white_intensity > 0.6
        # check if this is indeed keyboard image
