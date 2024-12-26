import sys
sys.path.append(r"C:\Users\orslu\Project_A")
from Project_A.RecognizeKeyboard.recognize_keyboard import KeyboardRecognizer

if __name__ == "__main__":
    keyboard_recognizer = KeyboardRecognizer()
    keyboard_recognizer.start()
