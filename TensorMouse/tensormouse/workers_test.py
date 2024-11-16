import cv2
import mediapipe as mp
import numpy as np
from multiprocessing import Value, Process
from pynput.keyboard import Listener, Key
from pynput.mouse import Button, Controller
import tkinter
import time

MOUSE_NULL, MOUSE_CLICK, MOUSE_DRAG, MOUSE_DRAGGING, MOUSE_RELEASE, QUIT = range(6)

def key_listener():
    return Listener(on_press=keypress_listener, on_release=keyrelease_listener)

def mouse_move_worker(handX, handY, mouse_state, is_clicking):
    mouse = Controller()
    x, y = tkinter.Tk().winfo_screenwidth(), tkinter.Tk().winfo_screenheight()

    while True:
        if mouse_state.value == MOUSE_CLICK or is_clicking.value:
            mouse.press(Button.left)
            time.sleep(0.1)
            mouse.release(Button.left)
            is_clicking.value = False
            mouse_state.value = MOUSE_NULL
        if mouse_state.value == MOUSE_DRAG:
            mouse.press(Button.left)
            mouse_state.value = MOUSE_DRAGGING
        if mouse_state.value == MOUSE_RELEASE:
            mouse.release(Button.left)
            mouse_state.value = MOUSE_NULL

        if handX.value >= 0 and handY.value >= 0:
            screen_x = int(handX.value * x)
            screen_y = int(handY.value * y)
            mouse.position = (screen_x, screen_y)

def keypress_listener(key):
    global mouse_state
    if key == Key.ctrl:
        mouse_state.value = MOUSE_CLICK
    elif key == Key.alt:
        mouse_state.value = MOUSE_DRAG
    elif key == Key.caps_lock:
        mouse_state.value = QUIT

def keyrelease_listener(key):
    global mouse_state
    if mouse_state.value == MOUSE_DRAGGING:
        mouse_state.value = MOUSE_RELEASE

def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def main_worker(camera_id):
    global mouse_state

    handX, handY = Value('d', 0.0), Value('d', 0.0)
    mouse_state = Value('i', MOUSE_NULL)
    is_clicking = Value('b', False)

    mouse_process = Process(target=mouse_move_worker, args=(handX, handY, mouse_state, is_clicking))
    mouse_process.start()

    keyboard_thread = key_listener()
    keyboard_thread.start()

    webcam = cv2.VideoCapture(camera_id)
    webcam.set(3, 640)
    webcam.set(4, 480)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(min_tracking_confidence=0.5, min_detection_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils

    print(u'\x1b[6;30;42m \u2713 TensorMouse with hand tracking started successfully! \033[0m')
    print("Use CTRL to perform clicks, ALT to cursor drag and press CAPS_LOCK to exit")
    print("Pinch your thumb and index finger together to click")

    while True:
        success, image = webcam.read()
        if not success:
            continue

        roi_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(roi_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)
                
                index_finger_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                
                # Normalize hand position within the full screen
                handX.value = max(0, min(1, index_finger_tip.x))
                handY.value = max(0, min(1, index_finger_tip.y))

                # print(f"mouse x = {handX.value}")
                # print(f"mouse y = {handY.value}")
                
                if calculate_distance(thumb_tip, index_finger_tip) < 0.05:
                    #! is_clicking.value = True 
                    cv2.putText(image, "Click!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(5) & 0xFF == 27 or mouse_state.value == QUIT:
            break

    mouse_process.terminate()
    keyboard_thread.stop()
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_worker(0)  # Use camera index 0 by default
