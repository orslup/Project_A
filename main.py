import cv2
import mediapipe as mp
import imutils


# Declaring MediaPipe objects
mpHands = mp.solutions.hands
hands = mpHands.Hands(min_tracking_confidence = 0.9)
mpDraw = mp.solutions.drawing_utils


# Processing the input image
def process_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(gray_image)
    # print(results.multi_hand_landmarks)
    return results


# Drawing landmark connections
def draw_hand_connections(img, results):
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                if id not in [4,8,12,16,20]:
                    continue
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy, lm.z)
                cv2.circle(img, (cx, cy), 8, (0, 255, 0),
                           cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        return img


def main():
    cap = cv2.VideoCapture(0)
    while True:
        success, image = cap.read()
        image = imutils.resize(image, width=800, height=600)
        results = process_image(image)
        draw_hand_connections(image, results)

        cv2.imshow("Hand tracker", image)

        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
