import cv2
import numpy as np
import argparse
from typing import Iterable, List, Tuple, Any, Dict, Union
import mediapipe as mp
import os
import glob
import sys


Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]
Image = np.ndarray

from Project_A.utils.keyboard_segmentation import KeyboardSegmentation
from Project_A.utils.hand_segmentation import HandSegmentation
from Project_A.utils.mouse_segmentation import MouseSegmentation
from Project_A.utils.keyboard_layout import Key, Keyboard_Layout

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


class DummyVideoCapture:
    def __init__(self, image_folder):
        self.image_paths = sorted(glob.glob(f"{image_folder}/*.png"))
        print(len(self.image_paths))
        self.index = 0

    def isOpened(self):
        return len(self.image_paths) > 0 and self.index < len(self.image_paths)

    def read(self):
        """Returns (status, frame) like cv2.VideoCapture.read()"""
        if self.index >= len(self.image_paths):
            return False, None
        frame = cv2.imread(self.image_paths[self.index])
        self.index += 1
        return True, frame

class KeyboardRecognizer:

    def __init__(self, settings: Union[Settings, None] = None, video_src_id: int = 0):
        if settings is None:
            self.settings = Settings()
        else:
            self.settings: Settings = settings
        self.video_src_id = video_src_id
        self.keyboard_segmentation = KeyboardSegmentation()
        self.hand_segmentation = HandSegmentation()
        self.mouse_hand_segmentation = HandSegmentation(history_size = 1)
        self.mouse_segmentation = MouseSegmentation()

    def start(self, save_dir=None) -> None:
        if isinstance(self.video_src_id, int) or os.path.isfile(self.video_src_id):
            cap = cv2.VideoCapture(self.video_src_id)
        else:
            cap = DummyVideoCapture(self.video_src_id)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        success = True
        frame_count = 0
        while success:
            success, image = cap.read()
            # image = cv2.flip(image, 1)
            # image = cv2.rotate(image, cv2.ROTATE_180)
            frame_count += 1
            if cap is None or not cap.isOpened() or not success:
                print("Video Capture Ended")
                sys.exit(0)
            if save_dir is not None:
                image_path = os.path.join(save_dir, f"frame_{frame_count:04d}.png")
                cv2.imwrite(image_path, image)
            self.segment_image(image)
            self.update_mouse_keyboard_state()
            if not self.settings.get_setting('hide_camera_image'):
                cv2.imshow("Hand tracker", image)
            keyboard_image = self.keyboard_segmentation.keyboard_image
            if not self.settings.get_setting('hide_keyboard_image') and keyboard_image is not None:
                cv2.imshow("Keyboard", keyboard_image)
            if self.settings.get_setting('activate_keyboard'):
                self.update_keyboard_click()
            if cv2.waitKey(1) == ord("m"):
                self.settings.toggle_setting('activate_mouse_movement')

            if cv2.waitKey(1) == ord("c"):
                self.settings.toggle_setting('activate_mouse_click')

            if cv2.waitKey(1) == ord("k"):
                self.settings.toggle_setting('activate_keyboard')

            if cv2.waitKey(1) == ord("q"):
                cap.release()
                cv2.destroyAllWindows()

    def segment_image(self, cam_image: Image) -> None:
        self.keyboard_segmentation.segment_keyboard(cam_image)
        self.hand_segmentation.segment_hands(cam_image,debug=False)
        self.mouse_hand_segmentation.segment_hands(cam_image)
        self.mouse_segmentation.segment_mouse()

    def update_mouse_keyboard_state(self) -> None:
        if self.mouse_hand_segmentation.identify_mouse_shape():
            if self.settings.get_setting('activate_mouse_click'):
                self.mouse_segmentation.update_mouse_state(self.mouse_hand_segmentation.identify_click())
            if self.settings.get_setting('activate_mouse_movement'):
                self.mouse_segmentation.mouse_move(self.mouse_hand_segmentation)
        
    def get_index_finger_on_keyboard(self) -> Point:
        try:
            return self.keyboard_segmentation.project_point(
                (self.hand_segmentation.index_finger[0],
                 self.hand_segmentation.index_finger[1])
            )
        except cv2.error as e:
            return self.keyboard_segmentation.NO_POINT

    def get_pressed_key(self) -> Union[Key, None]:
        # todo change to identify keyboard click
        if not self.hand_segmentation.identify_keyboard_click():
            return None
        key_point = self.get_index_finger_on_keyboard()
        if key_point == self.keyboard_segmentation.NO_POINT:
            return None
        keyboard_layout_obj = Keyboard_Layout((self.keyboard_segmentation.homography_width,
                                            self.keyboard_segmentation.homography_height))
        key = keyboard_layout_obj.get_key_by_index(*key_point)
        return key

    def update_keyboard_click(self) -> None:
        pressed_key = self.get_pressed_key()
        if pressed_key is None:
            return
        self.keyboard_segmentation.current_key = pressed_key.key_name
