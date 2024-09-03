import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import os

os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Function to extract frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Function to detect features in a frame
def detect_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
    if features is not None:
        features = np.float32(features)
    return features

# Function to track features between frames
def track_features(prev_frame, next_frame, prev_features):
    if prev_features is None:
        return None, None
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    next_features, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_features, None)
    return next_features, status

# Function to calculate optical flow between frames
def calculate_optical_flow(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Function to check texture and color consistency between frames
def texture_color_consistency(prev_frame, next_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, next_gray)
    return np.mean(diff)

# Function to detect morphing in a series of frames
def detect_morphing(frames):
    morphing_indices = []
    for i in range(len(frames) - 1):
        prev_frame = frames[i]
        next_frame = frames[i + 1]
        prev_features = detect_features(prev_frame)
        if prev_features is None:
            continue
        next_features, status = track_features(prev_frame, next_frame, prev_features)
        if next_features is None or status is None:
            continue
        flow = calculate_optical_flow(prev_frame, next_frame)
        consistency = texture_color_consistency(prev_frame, next_frame)
        
        # Define thresholds for detecting anomalies
        feature_threshold = 0.5
        flow_threshold = 1.0
        consistency_threshold = 20
        
        if np.mean(status) < feature_threshold or np.mean(flow) > flow_threshold or consistency > consistency_threshold:
            morphing_indices.append(i)
    
    return morphing_indices

# Function to visualize the morphing detection results
def visualize_morphing(frames, morphing_indices):
    for i in morphing_indices:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        plt.title('Frame ' + str(i))
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2RGB))
        plt.title('Frame ' + str(i + 1))
        plt.show()

# Function to load video and run the detection algorithm
def load_video():
    file_path = filedialog.askopenfilename()
    if not file_path:
        print("Video not found")
        return
    
    frames = extract_frames(file_path)
    morphing_indices = detect_morphing(frames)
    
    if morphing_indices:
        messagebox.showinfo("Morphing Detected", f"Morphing detected in frames: {morphing_indices}")

        visualize_morphing(frames, morphing_indices)
    else:
        messagebox.showinfo("No Morphing Detected", "No morphing detected in the video.")


# Create the main application window
root = tk.Tk()
root.title("Video Morphing Detection")
root.geometry("400x200")

# Create and place the "Load Video" button
load_button = tk.Button(root, text="Load Video", command=load_video)
load_button.pack(pady=20)

# Run the main event loop
root.mainloop()
