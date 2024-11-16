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
from Project_A.utils.mouse_segmentation import MouseSegmentation

class KeyboardRecognizer:

    def __init__(self):
        self.keyboard_segmentation = KeyboardSegmentation()
        self.hand_segmentation = HandSegmentation()
        self.mouse_segmentation = MouseSegmentation()

    def start(self) -> None:
        debug_flag = False
        cap = cv2.VideoCapture(0)
        success = True
        while success:
            success, image = cap.read()
            self.segment_image(image)
            self.update_mouse_keyboard_state()
            cv2.imshow("Hand tracker", image)
            keyboard_image = self.keyboard_segmentation.keyboard_image
            if keyboard_image is not None and debug_flag:
                cv2.imshow("Keyboard", keyboard_image)

            if cv2.waitKey(1) == ord("q"):
                cap.release()
                cv2.destroyAllWindows()

    def segment_image(self, cam_image: Image) -> None:
        self.keyboard_segmentation.segment_keyboard(cam_image)
        self.hand_segmentation.segment_hands(cam_image)
        self.mouse_segmentation.segment_mouse(self.hand_segmentation.index_finger[0],
                                              self.hand_segmentation.index_finger[1])

    def update_mouse_keyboard_state(self) -> None:
        self.mouse_segmentation.update_mouse_state(self.hand_segmentation.identify_click())
        self.mouse_segmentation.mouse_move(self.hand_segmentation.index_finger[0],
                                            self.hand_segmentation.index_finger[1])
        
    def get_index_finger_on_keyboard(self) -> Point:
        return self.keyboard_segmentation.project_point(
            self.hand_segmentation.index_finger
        )


def main():
    keyboard_recognizer = KeyboardRecognizer()
    keyboard_recognizer.start()


if __name__ == '__main__':
    main()
