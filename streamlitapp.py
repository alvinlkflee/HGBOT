#!/usr/bin/env python
# -*- coding: utf-8 -*-
import streamlit as st
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque, Counter
from utils import CvFpsCalc
from model import KeyPointClassifier, PointHistoryClassifier

def main():
    # Streamlit UI
    st.title("Hand Gesture Recognition App")
    st.sidebar.header("Configuration")
    
    device = st.sidebar.number_input("Camera Device Index", value=0, step=1)
    width = st.sidebar.slider("Frame Width", 640, 1920, 960)
    height = st.sidebar.slider("Frame Height", 480, 1080, 720)
    min_detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.7)
    min_tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.5)

    # Initialize Mediapipe Hand Detector
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Initialize models
    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    # Streamlit video capture
    run = st.checkbox("Run Hand Gesture Recognition")
    cap = cv.VideoCapture(device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

    # Display feed
    stframe = st.empty()
    fps_calc = CvFpsCalc(buffer_len=10)

    while run:
        fps = fps_calc.get()
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture video")
            break

        frame = cv.flip(frame, 1)
        debug_image = frame.copy()
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Process landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Gesture classification
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                # Drawing landmarks
                debug_image = draw_landmarks(debug_image, landmark_list)

                # Adding classification results
                cv.putText(debug_image, f"Sign ID: {hand_sign_id}", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        # Streamlit frame update
        stframe.image(cv.cvtColor(debug_image, cv.COLOR_BGR2RGB), channels="RGB")

    cap.release()

def calc_landmark_list(image, landmarks):
    height, width, _ = image.shape
    return [[int(landmark.x * width), int(landmark.y * height)] for landmark in landmarks.landmark]

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    normalized_landmarks = [[x - base_x, y - base_y] for x, y in landmark_list]
    max_value = max(max(abs(x) for x, y in normalized_landmarks), max(abs(y) for x, y in normalized_landmarks))
    return [coord / max_value for point in normalized_landmarks for coord in point]

def draw_landmarks(image, landmark_list):
    for point in landmark_list:
        cv.circle(image, tuple(point), 5, (0, 255, 0), -1)
    return image

if __name__ == "__main__":
    main()
