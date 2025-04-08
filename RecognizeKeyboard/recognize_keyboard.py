import cv2
import numpy as np
import argparse
from typing import Iterable, List, Tuple, Any, Dict, Union
import mediapipe as mp
import os
import glob
import sys
import threading
import time
from collections import deque

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


class CachedVideoCapture:
    def __init__(self, src=0, cache_size=64):
        self.cap = cv2.VideoCapture(src)
        self.cache = deque(maxlen=cache_size)
        self.lock = threading.Lock()
        self.stopped = False
        self.new_frame_event = threading.Event()

        # Start reading frames in background
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.cache.append(frame)
                        self.new_frame_event.set()  # Notify that a new frame is available
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.1)

    def read(self, timeout=None):
        # Wait until a frame is available
        while True:
            with self.lock:
                if self.cache:
                    return True, self.cache[-1].copy()
                else:
                    self.new_frame_event.clear()
            got_frame = self.new_frame_event.wait(timeout=timeout)
            if not got_frame:
                # Optional: add a timeout to prevent infinite blocking
                continue

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()


class KeyboardRecognizer:

    def __init__(self, settings: Union[Settings, None] = None, video_src_id: int = 0):
        if settings is None:
            self.settings = Settings()
        else:
            self.settings: Settings = settings
        self.video_src_id = video_src_id
        self.keyboard_segmentation = KeyboardSegmentation()
        self.hand_segmentation = HandSegmentation()
        self.mouse_hand_segmentation = HandSegmentation(history_size = 5,ignore_history = True)
        self.mouse_segmentation = MouseSegmentation()

    def start(self, save_dir=None) -> None:
        if isinstance(self.video_src_id, int) or os.path.isfile(self.video_src_id):
            cap = CachedVideoCapture(self.video_src_id)
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
            if cv2.waitKey(1) == ord("r"):
                self.keyboard_segmentation.reset_segmantation()
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
        if self.settings.get_setting('activate_keyboard'):
            self.keyboard_segmentation.segment_keyboard(cam_image)
            self.hand_segmentation.segment_hands(cam_image,debug=True)
        if self.settings.get_setting('activate_mouse_movement') or self.settings.get_setting('activate_mouse_click') :
            self.mouse_hand_segmentation.segment_hands(cam_image)

    def update_mouse_keyboard_state(self) -> None:
        hand_in_mouse_shape = self.mouse_hand_segmentation.identify_mouse_shape()
        if self.settings.get_setting('activate_mouse_click'):
            self.mouse_segmentation.update_mouse_state(self.mouse_hand_segmentation.identify_click_robust())
        if hand_in_mouse_shape:
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
        key_point = self.get_index_finger_on_keyboard()
        if not self.hand_segmentation.identify_click_robust():
            return None
        self.keyboard_segmentation.draw_click()
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
