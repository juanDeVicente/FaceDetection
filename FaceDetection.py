import cv2

display_width = 1600
display_height = 900

text_color = (255, 255, 255)

if __name__ == "__main__":

    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, display_width)
    video_capture.set(4, display_height)

    screen_range_w = display_width // len(funny_texts)
    screen_range_h = display_height // len(funny_texts)

    frontal_face_cascade = cv2.CascadeClassifier('xml_data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('xml_data/haarcascade_eye.xml')
    left_eye_cascade = cv2.CascadeClassifier('xml_data/haarcascade_lefteye_2splits.xml')
    right_eye_cascade = cv2.CascadeClassifier('xml_data/haarcascade_righteye_2splits.xml')
    profile_face_cascade = cv2.CascadeClassifier('xml_data/haarcascade_profileface.xml')

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = frontal_face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        faces = profile_face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Reconocimiento facial', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
