import os
import sys
import time
from statistics import mean

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
from absl import app, logging
from flask import Flask, Response, abort, jsonify, request, send_from_directory
from scipy.optimize import curve_fit

from .config import shooting_result
from .utils import detect_API, detect_image, detect_shot, openpose_init, tensorflow_init

tf.disable_v2_behavior()


def initialize_tensorflow():
    """Initializes TensorFlow with GPU settings."""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.36
    return config


def initialize_video_capture(video_path):
    """Initializes video capture and retrieves properties."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, width, height, fps


def setup_plot():
    """Sets up a matplotlib plot for trajectory visualization."""
    fig = plt.figure()
    plt.title("Trajectory Fitting", figure=fig)
    return fig


def save_output_images(fig, trace, height):
    """Saves the trajectory fitting and basketball trace images."""
    trajectory_path = os.path.join(
        os.getcwd(), "static/detections/trajectory_fitting.jpg"
    )
    plt.ylim(bottom=0, top=height)
    fig.savefig(trajectory_path)
    fig.clear()

    trace_path = os.path.join(os.getcwd(), "static/detections/basketball_trace.jpg")
    cv2.imwrite(trace_path, trace)


def calculate_avg_metrics(shooting_pose, during_shooting, fps):
    """Calculates average metrics from shooting data."""
    shooting_result["avg_elbow_angle"] = round(
        mean(shooting_pose["elbow_angle_list"]), 2
    )
    shooting_result["avg_knee_angle"] = round(mean(shooting_pose["knee_angle_list"]), 2)
    shooting_result["avg_release_angle"] = round(
        mean(during_shooting["release_angle_list"]), 2
    )
    shooting_result["avg_ballInHand_time"] = round(
        mean(shooting_pose["ballInHand_frames_list"]) * (4 / fps), 2
    )


def get_video_stream(video_path):
    """Processes video stream and performs detection."""
    # Initialize models and video capture
    datum, opWrapper = openpose_init()
    detection_graph, image_tensor, boxes, scores, classes, num_detections = (
        tensorflow_init()
    )
    cap, width, height, fps = initialize_video_capture(video_path)
    trace = np.full((height, width, 3), 255, np.uint8)
    fig = setup_plot()

    # Detection states
    previous = {
        "ball": np.array([0, 0]),
        "hoop": np.array([0, 0, 0, 0]),
        "hoop_height": 0,
    }
    during_shooting = {
        "isShooting": False,
        "balls_during_shooting": [],
        "release_angle_list": [],
        "release_point": [],
    }
    shooting_pose = {
        "ball_in_hand": False,
        "elbow_angle": 370,
        "knee_angle": 370,
        "ballInHand_frames": 0,
        "elbow_angle_list": [],
        "knee_angle_list": [],
        "ballInHand_frames_list": [],
    }
    shot_result = {"displayFrames": 0, "release_displayFrames": 0, "judgement": ""}
    skip_count = 0

    # TensorFlow session for detection
    with tf.Session(graph=detection_graph, config=initialize_tensorflow()) as sess:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            # Skip frames to optimize processing
            skip_count += 1
            if skip_count < 4:
                continue
            skip_count = 0

            detection, trace = detect_shot(
                img,
                trace,
                width,
                height,
                sess,
                image_tensor,
                boxes,
                scores,
                classes,
                num_detections,
                previous,
                during_shooting,
                shot_result,
                fig,
                datum,
                opWrapper,
                shooting_pose,
            )

            detection = cv2.resize(detection, (0, 0), fx=0.83, fy=0.83)
            frame = cv2.imencode(".jpg", detection)[1].tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    # Calculate and save metrics
    calculate_avg_metrics(shooting_pose, during_shooting, fps)
    save_output_images(fig, trace, height)


def get_image(image_path, img_name, response):
    """Processes a single image and saves detection output."""
    output_path = "./static/detections/"
    image = cv2.imread(image_path)
    detection = detect_image(image, response)
    cv2.imwrite(os.path.join(output_path, img_name), detection)
    print(f"Output saved to: {os.path.join(output_path, img_name)}")


def detection_api(response, image_path):
    """Processes detection via API for a given image."""
    image = cv2.imread(image_path)
    detect_API(response, image)
