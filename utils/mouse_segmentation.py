import tkinter, time
from pynput.mouse import Button, Controller

MOUSE_NULL = 0
MOUSE_CLICK = 1

from Project_A.utils.hand_segmentation import HandSegmentation

class MouseSegmentation(Controller):
    
    def __init__(self):
        super()
        self.mouse_state = MOUSE_NULL

    def update_mouse_state(self,is_mouse_click : bool):
        if is_mouse_click:
            self.mouse_state = MOUSE_CLICK
        else:
            self.mouse_state = MOUSE_NULL
        
    def mouse_move(self, mouse_hand_segmentation: HandSegmentation):
        x, y = tkinter.Tk().winfo_screenwidth(), tkinter.Tk().winfo_screenheight()

        # Define margin (in normalized units, e.g. 0.1 = 10%)
        margin = 0.1
        
        # Clamp the coordinates to the inner region (0.1 to 0.9)
        raw_x = mouse_hand_segmentation.index_finger[0]
        raw_y = mouse_hand_segmentation.index_finger[1]

        handX = max(margin, min(1 - margin, raw_x))
        handY = max(margin, min(1 - margin, raw_y))  

        # Re-normalize from margin–(1-margin) to 0–1
        norm_x = (handX - margin) / (1 - 2 * margin)
        norm_y = (handY - margin) / (1 - 2 * margin)      
        if self.mouse_state == MOUSE_CLICK:
            print("Mouse Click")
            # self.press(Button.left)
            # time.sleep(0.1)
            # self.release(Button.left)
            self.mouse_state = MOUSE_NULL

        if handX >= 0 and handY >= 0: #Normalize coordinates to screen size
            screen_x = int(norm_x * x)
            screen_y = int(norm_y * y)
            self.position = (screen_x, screen_y)
            # print(self.position)