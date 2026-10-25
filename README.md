# 🏏 Cricket Ball Tracker

An automated computer vision project that detects and tracks a cricket ball in video footage and visualizes its trajectory using Python, OpenCV, and a pretrained YOLO model.

## 📌 Project Overview

Cricket Ball Tracker aims to identify the cricket ball in match or practice videos and display its movement as a visual path. The project uses object detection and video processing techniques to help analyze ball movement during a delivery.

## ✨ Features

- **Automatic Ball Detection:** Uses a pretrained YOLO model to detect the ball.
- **Ball Tracking:** Tracks detected ball positions across video frames.
- **Trajectory Visualization:** Draws a colored path showing the detected ball movement.
- **Video Processing:** Processes input videos and saves annotated output videos.
- **Computer Vision:** Uses OpenCV and NumPy for image and video processing.

## 🛠️ Technologies Used

- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- PyTorch

## 🎯 Applications

- Cricket ball movement analysis
- Cricket practice and training
- Introduction to object detection and tracking
- Computer vision experimentation
- Sports video analysis

## 🚧 Limitations

The system relies on a general-purpose pretrained object detection model. Since a cricket ball can be very small and move quickly, detection may be inconsistent in some frames. The generated trajectory depends on the quality and continuity of the detected positions.

## 🔮 Future Improvements

- Improve detection of small, fast-moving cricket balls.
- Reduce false detections from unrelated objects.
- Improve tracking through occlusion and after the ball bounces.
- Estimate ball speed and trajectory.
- Add trajectory analysis and statistical visualization.

## 👩‍💻 Author

**Project Developer:** Nidhi

This project was developed to explore computer vision, object detection, and ball trajectory visualization using Python.


