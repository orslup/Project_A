import argparse
from typing import List, Dict, Tuple, Any
from Project_A.RecognizeKeyboard.recognize_keyboard import KeyboardRecognizer, Settings



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
                    prog='KeyboardMouseRecognizer',
                    description='Recognizes Keyboard and Mouse',
                    epilog='')
    parser.add_argument('-k', '--activate_keyboard', action='store_true')
    parser.add_argument('-mm', '--activate_mouse_movement', action='store_true')
    parser.add_argument('-mc', '--activate_mouse_click', action='store_true')
    parser.add_argument('-hk', '--hide_keyboard_image', action='store_true')
    parser.add_argument('-hm', '--hide_mouse_image', action='store_true')
    parser.add_argument('-hc', '--hide_camera_image', action='store_true')
    parser.add_argument('--id', default=0, type=int)
    parser.add_argument('--video-path', type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    settings = Settings.from_args(args)
    video_src_id = args.video_path or args.id
    keyboard_recognizer = KeyboardRecognizer(settings, video_src_id=video_src_id)
    keyboard_recognizer.start()
