from pathlib import Path

import cv2
from ultralytics import YOLO
from distort import add_gaussian_noise


model = YOLO("trained_s.pt")
DATA_DIR = Path("Data")

def process_video_no_distortion(video_index):
    return process_video(video_index, distort_frame=lambda frame: frame)

def process_video_with_blur(video_index):
    return process_video(video_index, distort_frame=lambda frame: cv2.blur(frame, (8, 8)))

def process_video_with_noise(video_index):
    return process_video(video_index, distort_frame=lambda frame: add_gaussian_noise(frame, 0, 35))

def process_video(video_index, distort_frame):
    video_path = f"{video_index}.MP4"
    output_path = f'out_{video_path}'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None, None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = 0
    recognised_frames = 0
    avg_conf = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = distort_frame(frame)

        results = model.track(frame, persist=True, conf=0.25, tracker="bytetrack.yaml")
        total_frames += 1

        annotated_frame = results[0].plot()

        if len(results[0].boxes) > 0:
            recognised_frames += 1
            avg_conf.extend(results[0].boxes.conf.cpu().numpy())

        out.write(annotated_frame)
        cv2.imshow("YOLO Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    percent_recognised = (recognised_frames / total_frames) if total_frames > 0 else 0
    avg_confidence = (sum(avg_conf) / len(avg_conf)) if avg_conf else 0

    return percent_recognised, avg_confidence

if __name__ == '__main__':
    percent = []
    conf = []

    percent_blur = []
    conf_blur = []

    percent_noise = []
    conf_noise = []

    for i in range(1, 7):
        percent_recognised, avg_confidence = process_video_no_distortion(i)
        percent_recognised_blur, avg_confidence_blur = process_video_with_blur(i)
        percent_recognised_noise, avg_confidence_noise = process_video_with_noise(i)

        percent.append(f"{percent_recognised:.2%}")
        conf.append(avg_confidence)

        percent_blur.append(f"{percent_recognised_blur:.2%}")
        conf_blur.append(avg_confidence_blur)

        percent_noise.append(f"{percent_recognised_noise:.2%}")
        conf_noise.append(avg_confidence_noise)

    print('No distortion')
    for p, c in zip(percent, conf):
        print(f"Recognition Rate: {p}, Average Confidence: {c:.2f}")

    print('Blur')
    for p, c in zip(percent_blur, conf_blur):
        print(f"Recognition Rate: {p}, Average Confidence: {c:.2f}")

    print('Noise')
    for p, c in zip(percent_noise, conf_noise):
        print(f"Recognition Rate: {p}, Average Confidence: {c:.2f}")
