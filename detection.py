import time
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
import os
import sys
import argparse
import matplotlib.pyplot as plt
from sys import platform
from scipy.optimize import curve_fit
from utils import openpose_init, tensorflow_init, detect_shot
from statistics import mean
tf.disable_v2_behavior()

datum, opWrapper = openpose_init()
detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()
frame_batch = 3

cap = cv2.VideoCapture("sample/one_score_one_miss.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("sample/output.avi", fourcc, fps / frame_batch, (int(width * 2 * 0.8), int(height * 0.8)))
trace = np.full((int(height), int(width), 3), 255, np.uint8)

fig = plt.figure()
#objects to store detection status
shooting_result = {
    'attempts': 0,
    'made': 0,
    'miss': 0,
    'avg_elbow_angle': 0,
    'avg_knee_angle': 0,
    'avg_release_angle': 0,
    'avg_ballInHand_time': 0
}
previous = {
'ball': np.array([0, 0]),  # x, y
'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
    'hoop_height': 0
}
during_shooting = {
    'isShooting': False,
    'balls_during_shooting': [],
    'release_angle_list': [],
    'release_point': []
}
shooting_pose = {
    'ball_in_hand': False,
    'elbow_angle': 370,
    'knee_angle': 370,
    'ballInHand_frames': 0,
    'elbow_angle_list': [],
    'knee_angle_list': [],
    'ballInHand_frames_list': []
}
shot_result = {
    'displayFrames': 0,
    'release_displayFrames': 0,
    'judgement': ""
}

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.38

skip_count = 0
with tf.Session(graph=detection_graph, config=config) as sess:
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        skip_count += 1
        if(skip_count < frame_batch):
            continue
        skip_count = 0
        detection, trace = detect_shot(img, trace, width, height, sess, image_tensor, boxes, scores, classes,
                                        num_detections, previous, during_shooting, shot_result, fig, shooting_result, datum, opWrapper, shooting_pose)

        detection = cv2.resize(detection, (0, 0), fx=0.8, fy=0.8)
        cv2.imshow("detection", detection)
        out.write(detection)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

# getting average shooting angle
shooting_result['avg_elbow_angle'] = round(mean(shooting_pose['elbow_angle_list']), 2)
shooting_result['avg_knee_angle'] = round(mean(shooting_pose['knee_angle_list']), 2)
shooting_result['avg_release_angle'] = round(mean(during_shooting['release_angle_list']), 2)
shooting_result['avg_ballInHand_time'] = round(mean(shooting_pose['ballInHand_frames_list']) * (frame_batch / fps), 2)

print("avg", shooting_result['avg_elbow_angle'])
print("avg", shooting_result['avg_knee_angle'])
print("avg", shooting_result['avg_release_angle'])
print("avg", shooting_result['avg_ballInHand_time'])

plt.title("Trajectory Fitting", figure=fig)
plt.ylim(bottom=0, top=height)
trajectory_path = os.path.join(os.getcwd(), "trajectory_fitting.jpg")
fig.savefig(trajectory_path)
fig.clear()
trace_path = os.path.join(os.getcwd(), "basketball_trace.jpg")
cv2.imwrite(trace_path, trace)
