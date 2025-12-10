import os
import cv2
import mediapipe as mp

def preprocess_img(img, face_detection, top_expand=0.95, bottom_expand=0.1, side_expand=0.35, blur_ksize=(100, 100)):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1, y1, w, h = int(bbox.xmin * W), int(bbox.ymin * H), int(bbox.width * W), int(bbox.height * H)

            x1_exp = max(0, int(x1 - w * side_expand))
            y1_exp = max(0, int(y1 - h * top_expand))
            w_exp = int(w * (1 + 2 * side_expand))
            h_exp = int(h * (1 + top_expand + bottom_expand))

            img[y1_exp:y1_exp + h_exp, x1_exp:x1_exp + w_exp] = cv2.blur(
                img[y1_exp:y1_exp + h_exp, x1_exp:x1_exp + w_exp], blur_ksize
            )

            cv2.rectangle(img, (x1_exp, y1_exp), (x1_exp + w_exp, y1_exp + h_exp), (0, 255, 0), 2)

    return img


output_dir = './output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

choice = input("Choose mode:\nA - Real-time webcam\nB - Video file\nEnter your choice (A/B): ").strip().lower()

mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if choice == "a":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = preprocess_img(frame, face_detection)
            cv2.imshow("Real-Time Face Blur", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Real-time session ended.")

    elif choice == "b":
        filePath = input("Enter video file path: ").strip()
        cap = cv2.VideoCapture(filePath)
        if not cap.isOpened():
            print(f"Error: Could not open video {filePath}")
            exit()

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read the first frame.")
            cap.release()
            exit()

        output_path = os.path.join(output_dir, 'javed karim.mp4')
        output_video = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'MP4V'),
            cap.get(cv2.CAP_PROP_FPS),
            (frame.shape[1], frame.shape[0])
        )

        while ret:
            frame = preprocess_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()
        print(f"Video processing complete. Saved to: {output_path}")

    else:
        print("Invalid choice. Please enter A or B.")
