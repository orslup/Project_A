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
        
        handX = max(0, min(1, mouse_hand_segmentation.index_finger[0]))
        handY = max(0, min(1, mouse_hand_segmentation.index_finger[1]))
        
        if self.mouse_state == MOUSE_CLICK:
            print("Mouse Click")
            # self.press(Button.left)
            # time.sleep(0.1)
            # self.release(Button.left)
            self.mouse_state = MOUSE_NULL

        if handX >= 0 and handY >= 0: #Normalize coordinates to screen size
            screen_x = int(handX * x)
            screen_y = int(handY * y)
            self.position = (screen_x, screen_y)
            # print(self.position)