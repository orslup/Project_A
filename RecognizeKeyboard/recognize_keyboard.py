import cv2
import numpy as np
import argparse
from typing import Iterable, List, Tuple, Any, Dict, Union
import mediapipe as mp


Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
Image = np.ndarray

from Project_A.utils.keyboard_segmentation import KeyboardSegmentation
from Project_A.utils.hand_segmentation import HandSegmentation
from Project_A.utils.mouse_segmentation import MouseSegmentation

class Settings:
    conf: Dict = {}

    def set_setting(self, name: str, default_value: Any):
        self.conf[name] = default_value
    
    def get_setting(self, name) -> Any:
        return self.conf[name]
    
    def toggle_setting(self, name: str):
        self.conf[name] = not self.conf[name]

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Settings':
        settings = cls()
        for arg in vars(args):
            settings.set_setting(arg, getattr(args, arg))
        return settings


class KeyboardRecognizer:

    def __init__(self, settings: Union[Settings, None] = None):
        if settings is None:
            self.settings = Settings()
        else:
            self.settings: Settings = settings
        self.keyboard_segmentation = KeyboardSegmentation()
        self.hand_segmentation = HandSegmentation()
        self.mouse_hand_segmentation = HandSegmentation(history_size = 5,ignore_history = True)
        self.mouse_segmentation = MouseSegmentation()

    def start(self) -> None:
        cap = cv2.VideoCapture(0)
        success = True
        while success:
            success, image = cap.read()
            self.segment_image(image)
            self.update_mouse_keyboard_state()
            if not self.settings.get_setting('hide_camera_image'):
                cv2.imshow("Hand tracker", image)
            keyboard_image = self.keyboard_segmentation.keyboard_image
            if not self.settings.get_setting('hide_keyboard_image') and keyboard_image is not None:
                cv2.imshow("Keyboard", keyboard_image)
            
            if cv2.waitKey(1) == ord("m"):
                self.settings.toggle_setting('activate_mouse_movement')
                if self.settings.get_setting("activate_mouse_movement"):
                    print("Activated mouse movement")

            if cv2.waitKey(1) == ord("c"):
                self.settings.toggle_setting('activate_mouse_click')
                if self.settings.get_setting("activate_mouse_click"):
                    print("Activated mouse click")

            if cv2.waitKey(1) == ord("k"):
                self.settings.toggle_setting('activate_keyboard')

            if cv2.waitKey(1) == ord("q"):
                cap.release()
                cv2.destroyAllWindows()

    def segment_image(self, cam_image: Image) -> None:
        self.keyboard_segmentation.segment_keyboard(cam_image)
        self.hand_segmentation.segment_hands(cam_image,debug=False)
        self.mouse_hand_segmentation.segment_hands(cam_image)

    def update_mouse_keyboard_state(self) -> None:
        hand_in_mouse_shape = self.mouse_hand_segmentation.identify_mouse_shape()
        if self.settings.get_setting('activate_mouse_click'):
            self.mouse_segmentation.update_mouse_state(self.mouse_hand_segmentation.identify_click())
        if hand_in_mouse_shape:
            if self.settings.get_setting('activate_mouse_movement'):
                self.mouse_segmentation.mouse_move(self.mouse_hand_segmentation)
        
    def get_index_finger_on_keyboard(self) -> Point:
        return self.keyboard_segmentation.project_point(
            self.hand_segmentation.index_finger
        )
