
from typing import List, Tuple, Dict, Union
from abc import ABC

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]

class Key():

    KEY_CODES: Dict[str, str] = {
        'Esc': '',

    }

    def __init__(self, key_name: str) -> None:
        self.key_name = key_name
        self.key_code: str = ""

    def simulate_key_press(self):
        """
        make the system think that the key was pressed
        """
        pass

    def simulate_key_release(self):
        """
        make the system think that the key was released
        """
        pass


class Keyboard_Layout:

    _DEFAULT_SIZE = (43, 43)

    BASE_IMAGE_SIZE = (734, 316)
    KEYS_LOCATION_DICT: Dict[Rect, str] = {
            (11, 11, *_DEFAULT_SIZE):  'Esc',
            (106, 11, *_DEFAULT_SIZE):  'F1',
            (154, 11, *_DEFAULT_SIZE):  'F2',
            (202, 11, *_DEFAULT_SIZE):  'F3',
            (250, 11, *_DEFAULT_SIZE):  'F4',
            (324, 11, *_DEFAULT_SIZE):  'F5',
            (371, 11, *_DEFAULT_SIZE):  'F6',
            (420, 11, *_DEFAULT_SIZE):  'F7',
            (470, 11, *_DEFAULT_SIZE):  'F8',
            (540, 11, *_DEFAULT_SIZE):  'F9',
            (586, 11, *_DEFAULT_SIZE):  'F10',
            (634, 11, *_DEFAULT_SIZE):  'F11',
            (683, 11, *_DEFAULT_SIZE):  'F12',
            (10, 75, *_DEFAULT_SIZE):  '~',
            (59, 75, *_DEFAULT_SIZE):  '1',
            (106, 75, *_DEFAULT_SIZE):  '2',
            (154, 75, *_DEFAULT_SIZE):  '3',
            (202, 75, *_DEFAULT_SIZE):  '4',
            (250, 75, *_DEFAULT_SIZE):  '5',
            (299, 75, *_DEFAULT_SIZE):  '6',
            (346, 75, *_DEFAULT_SIZE):  '7',
            (395, 75, *_DEFAULT_SIZE):  '8',
            (443, 75, *_DEFAULT_SIZE):  '9',
            (492, 75, *_DEFAULT_SIZE):  '0',
            (540, 75, *_DEFAULT_SIZE):  '-',
            (586, 75, *_DEFAULT_SIZE):  '+',
            (634, 75, *_DEFAULT_SIZE):  '\\',
            (683, 75, *_DEFAULT_SIZE):  'Backsapce',
            (11, 124, 61, 43):  'Tab',
            (78, 124, *_DEFAULT_SIZE):  'Q',
            (125, 124, *_DEFAULT_SIZE):  'W',
            (174, 124, *_DEFAULT_SIZE):  'E',
            (222, 124, *_DEFAULT_SIZE):  'R',
            (270, 124, *_DEFAULT_SIZE):  'T',
            (319, 124, *_DEFAULT_SIZE):  'Y',
            (367, 124, *_DEFAULT_SIZE):  'U',
            (413, 124, *_DEFAULT_SIZE):  'I',
            (463, 124, *_DEFAULT_SIZE):  'O',
            (510, 124, *_DEFAULT_SIZE):  'P',
            (558, 124, *_DEFAULT_SIZE):  '{',
            (606, 124, *_DEFAULT_SIZE):  '}',
            (652, 124, 68, 43):  'Enter',
            (11, 171, 79, 43):  'CapsLock',
            (96, 171, *_DEFAULT_SIZE):  'A',
            (146, 171, *_DEFAULT_SIZE):  'S',
            (192, 171, *_DEFAULT_SIZE):  'D',
            (240, 171, *_DEFAULT_SIZE):  'F',
            (289, 171, *_DEFAULT_SIZE):  'G',
            (337, 171, *_DEFAULT_SIZE):  'H',
            (385, 171, *_DEFAULT_SIZE):  'J',
            (433, 171, *_DEFAULT_SIZE):  'K',
            (481, 171, *_DEFAULT_SIZE):  'L',
            (529, 171, *_DEFAULT_SIZE):  ':',
            (576, 171, *_DEFAULT_SIZE):  '"',
            (625, 171, 98, 43):  'Enter',
            (11, 220, 109, 43):  'LeftShift',
            (129, 220, *_DEFAULT_SIZE):  'Z',
            (177, 220, *_DEFAULT_SIZE):  'X',
            (226, 220, *_DEFAULT_SIZE):  'C',
            (272, 220, *_DEFAULT_SIZE):  'V',
            (312, 220, *_DEFAULT_SIZE):  'B',
            (369, 220, *_DEFAULT_SIZE):  'N',
            (417, 220, *_DEFAULT_SIZE):  'M',
            (465, 220, *_DEFAULT_SIZE):  '<',
            (514, 220, *_DEFAULT_SIZE):  '>',
            (561, 220, *_DEFAULT_SIZE):  '?',
            (614, 220, 111, 43):  'RightShift',
            (11, 268, 60, 43):  'LeftCtrl',
            (79, 268, 50, 43):  'LeftWinKey',
            (132, 268, 50, 43):  'LeftAlt',
            (193, 268, 300, 43):  'Space',
            (499, 268, 50, 43):  'RightAlt',
            (555, 268, 50, 43):  'fn',
            (609, 268, 50, 43):  'RightWinKey',
            (664, 268, 60, 43):  'RightCtrl',
        }

    def __init__(self, image_size: Tuple[int, int], scale_factor: Tuple[float, float] = (1.0, 1.0)) -> None:
        self.image_size = image_size
        self.scale_factor = scale_factor
    
    @property
    def normalize_factor(self) -> Tuple[float, float]:
        return (self.BASE_IMAGE_SIZE[0] / self.image_size[0] * self.scale_factor[0],
        self.BASE_IMAGE_SIZE[1] / self.image_size[1] * self.scale_factor[1])
    
    @staticmethod
    def is_in_rect(x: float, y: float, rect: Rect):
        return x >= rect[0] and x < (rect[0] + rect[2]) and\
               y >= rect[1] and y < (rect[1] + rect[3])
    
    def get_key_by_index(self, x: int, y: int) -> Union[Key, None]:
        normalized_x, normalized_y = self.normalize_factor[0] * x, self.normalize_factor[1] * y
        # search for min index:
        matches = list(filter(lambda rect: self.is_in_rect(normalized_x, normalized_y, rect), self.KEYS_LOCATION_DICT.keys()))
        
        if len(matches) == 0:
            return None
        
        if len(matches) > 1:
            print("Multiple keys match for point. Redefine your keys configuration")
        
        match = matches[0]
        key_name = self.KEYS_LOCATION_DICT[match]
        return Key(key_name)

if __name__ == "__main__":
    layout = Keyboard_Layout(image_size=(736, 316))
    key = layout.get_key_by_index(690, 171)
    if key is None:
        print("Didn't found")
    else:
        print(key.key_name)
