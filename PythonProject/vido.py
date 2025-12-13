import cv2
import os

video_file = "/home/harribelsa/PycharmProjects/PythonProject/vidos/mid.mp4"
output_folder = "/home/harribelsa/PycharmProjects/PythonProject/training_data/medium"

os.makedirs(output_folder, exist_ok=True)

video_name = os.path.splitext(os.path.basename(video_file))[0]
print("Saving frames to:", os.path.abspath(output_folder))

cap = cv2.VideoCapture(video_file)

frame_count = 0
saved_count = 0
SAVE_EVERY = 3  # احفظ كل 3 فريمات

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames.")
        break

    if frame_count % SAVE_EVERY == 0:
        frame_filename = os.path.join(
            output_folder,
            f"{video_name}_frame_{saved_count:05d}.png"
        )
        cv2.imwrite(frame_filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Saved {saved_count} frames from video '{video_name}'")