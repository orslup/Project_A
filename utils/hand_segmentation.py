import cv2
import math
import numpy as np
import mediapipe as mp
from copy import deepcopy

from typing import Iterable, List, Tuple, Any

from Project_A.utils.history_queue import Queue

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
Image = np.ndarray

class HandSegmentation:

    NO_POINT = (-1, -1)
    MEAN_OVER = 3
    NO_CLICK = 0
    CLICKING = 1
    CLICKED = 2

    def __init__(self, history_size = 10, ignore_history = False) -> None:
        self.history_size = history_size
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.index_finger_tip: Point = self.NO_POINT
        self.thumb_tip: Point = self.NO_POINT
        self.results = None
        self.landmark_history = Queue(max_size = self.history_size)
        self.ignore_history = ignore_history
        self.keyboard_click_state = self.NO_CLICK

    def segment_hands(self, cam_image, debug=True) -> None:
        self._process_image(cam_image)
        if debug:
            self.draw_hand_connections(cam_image)
        try:
            self._update_fingers()
        except Exception as e:
            pass

    def _process_image(self, cam_image: Image) -> None:
        gray_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(gray_image)
        self.landmark_history.qpush(self.results)
        if not results.multi_hand_landmarks:
            return
        self.results = results

    def _get_hand_variance(self) -> Point:
        if self.results is None:
            return
        elif not self.results.multi_hand_landmarks:
            return self.NO_POINT
        for hand_landmarks in self.results.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        mean_x = sum([lm[0] for lm in landmarks]) / len(landmarks)
        mean_y = sum([lm[1] for lm in landmarks]) / len(landmarks)
        mean_z = sum([lm[2] for lm in landmarks]) / len(landmarks)
        variance_x = sum([(lm[0] - mean_x) ** 2 for lm in landmarks]) / len(landmarks)
        variance_y = sum([(lm[1] - mean_y) ** 2 for lm in landmarks]) / len(landmarks)
        variance_z = sum([(lm[2] - mean_z) ** 2 for lm in landmarks]) / len(landmarks)
        return (variance_x, variance_y, variance_z)

    def _calculate_mean_landmarks(self, over_last=-1):
        if self.ignore_history:
            over_last = 1
        elif over_last == -1:
            over_last = self.HISTORY_SIZE
            
        if self.landmark_history.qrealsize() == 0:
            return
        
        landmarks = [[(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark] for results in self.landmark_history.qfiltered()[-over_last:]]
        all_landmarks = np.array(landmarks)
        mean_landmarks = np.mean(all_landmarks, axis=0)
        mean_result = deepcopy(self.landmark_history.qpeek())
        for i, (x, y, z) in enumerate(mean_landmarks):
            mean_result.multi_hand_landmarks[0].landmark[i].x = x
            mean_result.multi_hand_landmarks[0].landmark[i].y = y
            mean_result.multi_hand_landmarks[0].landmark[i].z = z
        return mean_result

    def draw_hand_connections(self, cam_image: Image) -> Image:
        if self.landmark_history.qrealsize() == 0:
            return
        mean_landmarks_result = self._calculate_mean_landmarks(self.MEAN_OVER)
        for handLms in mean_landmarks_result.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = cam_image.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                if id == 8:
                    pass
                    # print(cx, cy, cz)
                cv2.circle(cam_image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
            self.mpDraw.draw_landmarks(cam_image, handLms, self.mpHands.HAND_CONNECTIONS)
        return cam_image

    def _update_fingers(self) -> None:
        if self.results is None:
            return
        # Get index finger tip and thumb tip positions
        for hand_landmarks in self.results.multi_hand_landmarks:
            self.index_finger_tip = (hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y)
            self.thumb_tip = (hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y)
            
            self.wrist_landmark = hand_landmarks.landmark[0]
            self.thumb_tip_landmark = hand_landmarks.landmark[4]
            self.index_knuckle_landmark = hand_landmarks.landmark[5]
            self.index_finger_tip_landmark = hand_landmarks.landmark[8]
            self.middle_finger_tip_landmark = hand_landmarks.landmark[12]
            break
    
    @staticmethod
    def mean_point(p1, p2) -> Tuple[float, float, float]:
        return (p1.x + p2.x) / 2, (p1.y + p2.y) / 2, (p1.z + p2.z) / 2

    @staticmethod
    def euclidean_distance(p1, p2):
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

    def identify_click(self):
         # Ensure there are enough frames in the history for comparison
        if self.landmark_history.qrealsize() < 4:
            return False
        # Get the last four positions of the finger landmarks
        recent_landmarks = self.landmark_history.qfiltered()[-4:]
        # --- Fingers motion check ---
        index_means, pinky_means, wrist_means, pinky_mcp_means = [
            [
                self.mean_point(recent_landmarks[i].multi_hand_landmarks[0].landmark[finger_index],
                                recent_landmarks[i+1].multi_hand_landmarks[0].landmark[finger_index])
                for i in range(3)
            ]
            for finger_index in (8, 20, 0, 17)
        ]
        # index_means, pinky_means, wrist_means, pinky_mcp_means = [
        #     [
        #         self.mean_point(recent_landmarks[i+1].multi_hand_landmarks[0].landmark[finger_index],
        #                         recent_landmarks[i+1].multi_hand_landmarks[0].landmark[finger_index])
        #         for i in range(3)
        #     ]
        #     for finger_index in (8, 20, 0, 17)
        # ]

        index_movements, pinky_movements, wrist_movements, pinky_mcp_movements = [
            [
                self.euclidean_distance(finger_means[i], finger_means[i+1])
                for i in range(2)
            ]
            for finger_means in (index_means, pinky_means, wrist_means, pinky_mcp_means)
        ]

        # Check if there's a "rapid" movement (tune threshold as needed)
        fast_index_motion = any(dist > 0.05 for dist in index_movements)
        stable_pinky = any(dist < 0.06 for dist in pinky_movements)
        stable_wrist = any(dist < 0.06 for dist in wrist_movements)
        stable_pinky_mcp = any(dist < 0.06 for dist in pinky_mcp_movements)

        # fast_index_th = max(dist for dist in index_movements)
        # stable_pinky_th = min(dist for dist in pinky_movements)
        # stable_wrist_th = min(dist for dist in wrist_movements)
        # stable_pinky_mcp_th = min(dist for dist in pinky_mcp_movements)

        # Click happens if index moved fast, and other fingers stayed still
        # print(f"{fast_index_motion=} {stable_pinky_th=:.2} {stable_wrist_th=:.2} {stable_pinky_mcp_th=:.2}")
        # print(f"{fast_index_th=:.2}")

        # reset state if click was over
        if self.keyboard_click_state == self.CLICKED:
            self.keyboard_click_state = self.NO_CLICK

        # update clicking state
        if self.keyboard_click_state in (self.NO_CLICK, self.CLICKING):
            if fast_index_motion and (stable_pinky + stable_wrist + stable_pinky_mcp > 2):
                self.keyboard_click_state = self.CLICKING
            elif self.keyboard_click_state == self.CLICKING:
                self.keyboard_click_state = self.CLICKED
        return self.keyboard_click_state == self.CLICKED

    def identify_click_old(self) -> bool:
        """Identify mouse click

        implementation options:
            - index finger rapid location change
            
        Returns:
            bool: Mouse click occured in recent frames
        """
        # Ensure there are enough frames in the history for comparison
        if self.landmark_history.qrealsize() < 3:
            return False

        # Get the last three positions of the index finger tip
        recent_landmarks = self.landmark_history.qfiltered()[-3:]
        try:
            p1 = recent_landmarks[0].multi_hand_landmarks[0].landmark[8]  # Earlier frame
            p2 = recent_landmarks[1].multi_hand_landmarks[0].landmark[8]  # Mid frame
            p3 = recent_landmarks[2].multi_hand_landmarks[0].landmark[8]  # Latest frame
        except (AttributeError, IndexError):
            return False  # No valid landmarks to compare

        # Calculate distances between consecutive points
        d1 = self.calculate_distance(p1, p2)
        d2 = self.calculate_distance(p2, p3)

        # Define thresholds for localized rapid movement
        localized_threshold = 0.05  # Adjust for scale (normalized coordinates)
        total_distance_threshold = 0.1  # Larger threshold for overall movement
        #print(f"d1 = {d1}, d2 = {d2}")
        # Check if the movement is rapid but localized
        if d1 < localized_threshold and d2 < localized_threshold and (d1 + d2) > localized_threshold:
            print("Click detected")
            return True  # Rapid local movement detected
        elif (d1 + d2) > total_distance_threshold:
            return False  # Ignore large movements across the screen
        return False
    
    def identify_keyboard_click(self) -> bool:
        landmarks = [[(lm.x, lm.y, lm.z) for lm in results.multi_hand_landmarks[0].landmark] for results in self.landmark_history.qfiltered()]
        if len(landmarks) < 2:
            return False
        # return True
        i = np.array(landmarks[0])
        total_movement = 0.0
        for j in landmarks[1:]:
            j = np.array(j)
            total_movement += float(np.sum(np.abs(j - i)))
        print(total_movement)
        return total_movement < 8

    def identify_mouse_shape(self) -> bool:
        """Identify if hand is in mouse shape for movement 
        Returns:
            bool: Hand is in mouse shape
        """
        if self.results is None:
            return
        reference_distance = self.calculate_distance(self.wrist_landmark, self.middle_finger_tip_landmark)
        thumb_index_distance = self.calculate_distance(self.thumb_tip_landmark, self.index_finger_tip_landmark)
        normalized_thumb_index_distance = self.normalize_distance(thumb_index_distance, reference_distance)

        index_curvature_angle = self.calculate_angle(self.index_finger_tip_landmark,
                                                     self.index_knuckle_landmark,
                                                     self.wrist_landmark)

        # Print for debugging
        # print(f"Normalized Thumb-Index Distance: {normalized_thumb_index_distance:.2f}")
        # print(f"Index Finger Curvature Angle: {math.degrees(index_curvature_angle):.2f}°")

        if normalized_thumb_index_distance < 0.3 and index_curvature_angle > 1.0:
            return True
        else:
            return False

    @staticmethod
    def calculate_distance(p1, p2) -> float:
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)**0.5
    
    @staticmethod
    def normalize_distance(distance, reference_distance) -> float:
        if reference_distance == 0:
            return 0
        return distance / reference_distance
    
    @staticmethod
    def calculate_angle(p1, p2, p3) -> float:
        """
        Calculates the angle between vectors formed by three points.
        :param p1: First point.
        :param p2: Middle point (vertex).
        :param p3: Third point.
        :return: Angle in radians.
        """
        v1 = [p1.x - p2.x, p1.y - p2.y, p1.z - p2.z]
        v2 = [p3.x - p2.x, p3.y - p2.y, p3.z - p2.z]
        
        # Dot product of v1 and v2
        dot_product = sum(v1[i] * v2[i] for i in range(3))
        
        # Norm (magnitude) of vectors
        norm_v1 = math.sqrt(sum(v1[i]**2 for i in range(3)))
        norm_v2 = math.sqrt(sum(v2[i]**2 for i in range(3)))
        
        # Avoid division by zero
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        
        # Angle calculation
        angle = math.acos(dot_product / (norm_v1 * norm_v2))
        return angle

    @property
    def index_finger(self) -> Point:
        return self.index_finger_tip

    @property
    def thumb_finger(self) -> Point:
        return self.thumb_tip
