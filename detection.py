import time
from absl import app, logging
import cv2
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request, Response, jsonify, send_from_directory, abort
import os
import sys
from sys import platform
import argparse
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
tf.disable_v2_behavior()


def fit_func(x, a, b, c):
    return a*(x ** 2) + b * x + c


def trajectory_fit(balls, height, width, shotJudgement, fig):
    x = []
    y = []
    for ball in balls:
        x.append(ball[0])
        y.append(height - ball[1])

    try:
        params = curve_fit(fit_func, x, y)
        [a, b, c] = params[0]
    except:
        print("fiiting error")
        a = 0
        b = 0
        c = 0
    x_pos = np.arange(0, width, 1)
    y_pos = []
    for i in range(len(x_pos)):
        x_val = x_pos[i]
        y_val = (a * (x_val ** 2)) + (b * x_val) + c
        y_pos.append(y_val)

    if(shotJudgement == "MISS"):
        plt.plot(x, y, 'ro', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='red',
                 alpha=0.4, linewidth=5, figure=fig)
    else:
        plt.plot(x, y, 'go', figure=fig)
        plt.plot(x_pos, y_pos, linestyle='-', color='green',
                 alpha=0.4, linewidth=5, figure=fig)


def distance(xCoor, yCoor, prev_ball):
    return ((prev_ball[0] - xCoor) ** 2 + (prev_ball[1] - yCoor) ** 2) ** (1/2)

def tensorflow_init():
    MODEL_NAME = 'inference_graph'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    return detection_graph, image_tensor, boxes, scores, classes, num_detections


def detect_shot(frame, trace, width, height, sess, image_tensor, boxes, scores, classes, num_detections, previous, during_shooting, shot_result, fig, shooting_result):
    if(shot_result['displayFrames'] > 0):
        shot_result['displayFrames'] -= 1
    frame_expanded = np.expand_dims(frame, axis=0)
    # main tensorflow detection
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            if(classes[0][i] == 1):  # Basketball
                # During Shooting
                if(ymin < (previous['hoop_height'])):
                    if(not during_shooting['isShooting']):
                        during_shooting['isShooting'] = True

                    during_shooting['balls_during_shooting'].append(
                        [xCoor, yCoor])

                    #draw purple circle
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=10,
                               color=(235, 103, 193), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=10,
                               color=(235, 103, 193), thickness=-1)

                # Not shooting, and avoid misdetecting head as ball
                elif(ymin >= (previous['hoop_height'] - 30) and (distance(xCoor, yCoor, previous['ball']) < 100)):
                    # the moment when ball go below basket
                    if(during_shooting['isShooting']):
                        if(xCoor >= previous['hoop'][0] and xCoor <= previous['hoop'][2]):  # shot
                            shooting_result['attempts'] += 1
                            shooting_result['made'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "SCORE"
                            print("SCORE")
                            # draw green trace when miss
                            for ballCoor in during_shooting['balls_during_shooting']:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(82, 168, 50), thickness=-1)
                        else:  # miss
                            shooting_result['attempts'] += 1
                            shooting_result['miss'] += 1
                            shot_result['displayFrames'] = 10
                            shot_result['judgement'] = "MISS"
                            print("miss")
                            # draw red trace when miss
                            for ballCoor in during_shooting['balls_during_shooting']:
                                cv2.circle(img=trace, center=(ballCoor[0], ballCoor[1]), radius=10,
                                           color=(0, 0, 255), thickness=-1)
                        trajectory_fit(
                            during_shooting['balls_during_shooting'], height, width, shot_result['judgement'], fig)
                        during_shooting['balls_during_shooting'].clear()
                        during_shooting['isShooting'] = False

                    #draw blue circle
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=10,
                               color=(255, 0, 0), thickness=-1)

                previous['ball'][0] = xCoor
                previous['ball'][1] = yCoor

            if(classes[0][i] == 2):  # Rim
                # cover previous hoop with white rectangle
                cv2.rectangle(
                    trace, (previous['hoop'][0], previous['hoop'][1]), (previous['hoop'][2], previous['hoop'][3]), (255, 255, 255), 5)
                cv2.rectangle(frame, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)
                cv2.rectangle(trace, (xmin, ymax),
                              (xmax, ymin), (48, 124, 255), 5)

                #display judgement after shot
                if(shot_result['displayFrames']):
                    if(shot_result['judgement'] == "MISS"):
                        cv2.putText(frame, shot_result['judgement'], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 8)
                    else:
                        cv2.putText(frame, shot_result['judgement'], (xCoor - 65, yCoor - 65),
                                    cv2.FONT_HERSHEY_COMPLEX, 3, (82, 168, 50), 8)

                previous['hoop'][0] = xmin
                previous['hoop'][1] = ymax
                previous['hoop'][2] = xmax
                previous['hoop'][3] = ymin
                if(ymin > previous['hoop_height']):
                    previous['hoop_height'] = ymin
    combined = np.concatenate((frame, trace), axis=1)
    return combined, trace




detection_graph, image_tensor, boxes, scores, classes, num_detections = tensorflow_init()

cap = cv2.VideoCapture("sample/one_score_one_miss.mp4")
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
trace = np.full((int(height), int(width), 3), 255, np.uint8)

fig = plt.figure()
#objects to store detection status
shooting_result = {
    "attempts": 0,
    "made": 0,
    "miss": 0
}
previous = {
'ball': np.array([0, 0]),  # x, y
'hoop': np.array([0, 0, 0, 0]),  # xmin, ymax, xmax, ymin
    'hoop_height': 0
}
during_shooting = {
    'isShooting': np.array([False]),
    'balls_during_shooting': []
    }
shot_result = {
    'displayFrames': 0,
    'judgement': ""
    }

skip_count = 0
with tf.Session(graph=detection_graph) as sess:
    while True:
        ret, img = cap.read()
        skip_count += 1
        if(skip_count == 2):
            skip_count = 0
            continue
        if ret == False:
            break
        detection, trace = detect_shot(img, trace, width, height, sess, image_tensor, boxes, scores, classes,
                                        num_detections, previous, during_shooting, shot_result, fig, shooting_result)

        detection = cv2.resize(detection, (0, 0), fx=0.8, fy=0.8)
        cv2.imshow("detection", detection)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()

# plt.title("Trajectory Fitting", figure=fig)
# plt.ylim(bottom=0, top=height)
# trajectory_path = os.path.join(
#     os.getcwd(), "static/detections/trajectory_fitting.jpg")
# fig.savefig(trajectory_path)
# fig.clear()
# trace_path = os.path.join(
#     os.getcwd(), "static/detections/basketball_trace.jpg")
# cv2.imwrite(trace_path, trace)
