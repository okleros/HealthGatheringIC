import cv2
import os

# Path to the folder containing frames
frame_folder = "gradcam_output_highlighted_frames"

# Get all frames sorted by filename
frames = sorted(os.listdir(frame_folder))

# Read the first frame to get dimensions
first_frame = cv2.imread(os.path.join(frame_folder, frames[0]))
height, width, layers = first_frame.shape

# Define codec and create VideoWriter object
out = cv2.VideoWriter("videos/output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))

# Loop through frames and write to video
for frame in frames:
    img = cv2.imread(os.path.join(frame_folder, frame))
    out.write(img)

out.release()
cv2.destroyAllWindows()