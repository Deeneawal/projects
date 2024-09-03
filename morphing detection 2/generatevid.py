import cv2
import numpy as np

def create_morphing_video(output_path, width=640, height=480, fps=30, duration=5):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(fps * duration):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Define morphing factor
        morph_factor = i / (fps * duration)

        # Draw a rectangle and a circle with morphing effect
        if morph_factor < 0.5:
            cv2.rectangle(frame, (int(100 * morph_factor), int(100 * morph_factor)),
                          (width - int(100 * morph_factor), height - int(100 * morph_factor)), (0, 255, 0), -1)
        else:
            cv2.circle(frame, (width // 2, height // 2), int(200 * (1 - morph_factor)), (0, 0, 255), -1)

        out.write(frame)

    out.release()

# Create and save the test video
create_morphing_video('test_morphing_video.mp4')
