import tkinter, time
from pynput.mouse import Button, Controller

MOUSE_NULL = 0
MOUSE_CLICK = 1

class MouseSegmentation(Controller):
    
    def __init__(self):
        super()
        self.mouse_state = MOUSE_NULL
    
    def segment_mouse(self, handX, handY):
        """Identify if hand movement is valid for mouse movement 
        (for example hand shape for mouse, if hand is in ROI)
        """
        pass

    def update_mouse_state(self,is_mouse_click):
        if is_mouse_click:
            self.mouse_state = MOUSE_CLICK
        else:
            self.mouse_state = MOUSE_NULL
        
    def mouse_move(self,handX, handY):
        x, y = tkinter.Tk().winfo_screenwidth(), tkinter.Tk().winfo_screenheight()

        handX = max(0, min(1, handX))
        handY = max(0, min(1, handY))
        
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