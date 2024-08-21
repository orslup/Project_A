import cv2
import numpy as np
import mediapipe as mp
from copy import deepcopy

from typing import Iterable, List, Tuple, Any

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
Image = np.ndarray

from Project_A.utils.keyboard_segmentation import KeyboardSegmentation
from Project_A.utils.hand_segmentation import HandSegmentation

class KeyboardRecognizer:

    def __init__(self):
        self.keyboard_segmentation = KeyboardSegmentation()
        self.hand_segmentation = HandSegmentation()

    def start(self) -> None:
        cap = cv2.VideoCapture(0)
        success = True
        while success:
            success, image = cap.read()
            self.segment_image(image)
            cv2.imshow("Hand tracker", image)
            keyboard_image = self.keyboard_segmentation.keyboard_image
            if keyboard_image is not None:
                cv2.imshow("Keyboard", keyboard_image)

            if cv2.waitKey(1) == ord("q"):
                cap.release()
                cv2.destroyAllWindows()

    def segment_image(self, cam_image: Image) -> None:
        self.keyboard_segmentation.segment_keyboard(cam_image)
        self.hand_segmentation.segment_hands(cam_image)

    def is_key_press(self) -> bool:
        pass

    def get_index_finger_on_keyboard(self) -> Point:
        return self.keyboard_segmentation.project_point(
            self.hand_segmentation.index_finger
        )


def main():
    keyboard_recognizer = KeyboardRecognizer()
    keyboard_recognizer.start()


if __name__ == '__main__':
    main()
