import gradio as gr
from gradio_webrtc import WebRTC
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque
import itertools
import csv
from model import KeyPointClassifier, PointHistoryClassifier

# Load models and labels
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

keypoint_classifier = KeyPointClassifier()
point_history_classifier = PointHistoryClassifier()

with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = [row[0] for row in csv.reader(f)]

with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
    point_history_classifier_labels = [row[0] for row in csv.reader(f)]

# Initialize variables
point_history = deque(maxlen=16)

# Functions for hand processing
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def draw_landmarks(image, landmark_point):
    for index, point in enumerate(landmark_point):
        cv.circle(image, tuple(point), 5, (0, 255, 0), -1)
    return image

def process_frame(image):
    global point_history

    # Convert BGR to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = calc_landmark_list(image, hand_landmarks)
            pre_processed_landmark_list = itertools.chain.from_iterable(
                [[x - landmark_list[0][0], y - landmark_list[0][1]] for x, y in landmark_list]
            )
            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

            if hand_sign_id == 2:  # Example: Point gesture
                point_history.append(landmark_list[8])
            else:
                point_history.append([0, 0])

            image = draw_landmarks(image, landmark_list)

    return image

def detection(image, conf_threshold):
    # Process each frame from the WebRTC stream
    processed_image = process_frame(image)
    return processed_image

# Define WebRTC settings
rtc_configuration = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

# Define Gradio UI with WebRTC
css = """.my-group {max-width: 600px !important; max-height: 600px !important;}
         .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        Hand Gesture Recognition (Powered by WebRTC ⚡️)
        </h1>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="Stream", rtc_configuration=rtc_configuration)
            conf_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.0,
                maximum=1.0,
                step=0.05,
                value=0.30,
            )

        image.stream(
            fn=detection, inputs=[image, conf_threshold], outputs=image, time_limit=10
        )

if __name__ == "__main__":
    demo.launch()
