import cv2
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
    HISTORY_SIZE = 10
    MEAN_OVER = 3

    def __init__(self) -> None:
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.index_finger_tip: Point = self.NO_POINT
        self.thumb_tip: Point = self.NO_POINT
        self.results = None
        self.landmark_history = Queue(max_size=self.HISTORY_SIZE)

    def segment_hands(self, cam_image, debug=True) -> None:
        self._process_image(cam_image)
        if debug:
            self.draw_hand_connections(cam_image)
        try:
            self._update_fingers()
        except:
            return

    def _process_image(self, cam_image: Image) -> None:
        gray_image = cv2.cvtColor(cam_image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(gray_image)
        self.landmark_history.qpush(self.results)
        if not results.multi_hand_landmarks:
            return
        self.results = results

    def _get_hand_variance(self) -> Point:
        if not self.results.multi_hand_landmarks:
            return self.NO_POINT
        for hand_landmarks in self.results.multi_hand_landmarks:
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        mean_x = sum([lm[0] for lm in landmarks]) / len(landmarks)
        mean_y = sum([lm[1] for lm in landmarks]) / len(landmarks)
        variance_x = sum([(lm[0] - mean_x) ** 2 for lm in landmarks]) / len(landmarks)
        variance_y = sum([(lm[1] - mean_y) ** 2 for lm in landmarks]) / len(landmarks)
        return (variance_x, variance_y)

    def _calculate_mean_landmarks(self, over_last=-1):
        if over_last == -1:
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
        print(self.landmark_history.qrealsize())
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
        self.index_finger_tip = self.results.multi_hand_landmarks.landmark[8]
        self.thumb_tip = self.results.landmark[4]

    @property
    def index_finger(self) -> Point:
        return self.index_finger_tip

    @property
    def thumb_finger(self) -> Point:
        return self.thumb_tip
