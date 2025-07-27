# SignSense: Real-Time Sign Language Recognition

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**SignSense** is a complete, end-to-end machine learning project that detects and translates American Sign Language (ASL) gestures from a live webcam feed. This project demonstrates a full machine learning pipeline, from custom data collection and preprocessing to model training and real-time deployment.

---

### ✨ Core Features

*   **Real-Time Gesture Recognition**: Identifies a vocabulary of ASL signs (`hello`, `thanks`, `iloveyou`, etc.) directly from a webcam stream.
*   **Custom Data Pipeline**: The entire project is built on a custom-collected dataset, demonstrating the full ML lifecycle.
*   **Temporal Action Detection**: Utilizes an LSTM (Long Short-Term Memory) neural network to understand gestures as sequences of motion over time, not just static images.
*   **High-Performance Landmark Detection**: Leverages Google's MediaPipe framework for fast and accurate extraction of hand and body keypoints.

---

### 🛠️ Tech Stack & Pipeline

This project is built with industry-standard libraries for computer vision and deep learning. The pipeline is structured as follows:

1.  **Keypoint Extraction (`MediaPipe`)**:
    *   The `Holistic` model from MediaPipe is used to extract 3D coordinates for keypoints on the hands and body from each video frame. This converts video data into a numerical format.

2.  **Data Collection & Labeling (`OpenCV` & `NumPy`)**:
    *   A custom script uses OpenCV to capture sequences of video frames for each sign.
    *   The extracted keypoints for each sequence are saved as NumPy arrays, creating a labeled dataset for training.

3.  **Model Architecture (`TensorFlow` & `Keras`)**:
    *   A **Sequential** model is built using Keras.
    *   **LSTM (Long Short-Term Memory)** layers are used to process the sequence of keypoints, allowing the model to learn temporal patterns specific to each sign.
    *   **Dense** layers and a `softmax` activation function are used for the final classification of the sign.

4.  **Training & Evaluation (`Scikit-learn`)**:
    *   The model is trained on the collected data.
    *   Performance is evaluated using a **Confusion Matrix** and **Accuracy Score** to ensure reliability.

5.  **Real-Time Inference (`OpenCV` & `TensorFlow`)**:
    *   The trained model is loaded and runs on a live webcam feed.
    *   Keypoints are extracted from the live feed, fed into the model for prediction, and the recognized sign is displayed on the screen.

