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

def mouse_move_worker(handX, handY, mouse_state):
    mouse = Controller()
    x, y = tkinter.Tk().winfo_screenwidth(), tkinter.Tk().winfo_screenheight()

    while True:
        if mouse_state.value == MOUSE_CLICK:
            time.sleep(0.2)
            mouse.press(Button.left)
            mouse_state.value = MOUSE_RELEASE
        if mouse_state.value == MOUSE_DRAG:
            time.sleep(0.2)
            mouse.press(Button.left)
            mouse_state.value = MOUSE_DRAGGING
        if mouse_state.value == MOUSE_RELEASE:
            mouse.release(Button.left)
            mouse_state.value = MOUSE_NULL

        if handX.value > 0 and handY.value > 0:
            mouse.position = (int((1-handX.value)*x), int(handY.value*y))

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

def main_worker(camera_id):
    global mouse_state

    handX, handY = Value('d', 0.0), Value('d', 0.0)
    mouse_state = Value('i', MOUSE_NULL)

    mouse_process = Process(target=mouse_move_worker, args=(handX, handY, mouse_state))
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

    while True:
        success, image = webcam.read()
        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(image, hand_landmarks, mpHands.HAND_CONNECTIONS)
                index_finger_tip = hand_landmarks.landmark[8]
                handX.value, handY.value = index_finger_tip.x, index_finger_tip.y

        cv2.imshow("Hand Tracking", image)
        if cv2.waitKey(5) & 0xFF == 27 or mouse_state.value == QUIT:
            break

    mouse_process.terminate()
    keyboard_thread.stop()
    webcam.release()
    cv2.destroyAllWindows()