import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import time
import os
import sys
from sys import platform
import argparse
from utils import label_map_util
from utils import visualization_utils as vis_util
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
tf.disable_v2_behavior()
dir_path = 'C:/Users/tcheo/Desktop/Git/openpose/build/examples/tutorial_api_python'

try:
    if platform == "win32":
        print(dir_path)
        sys.path.append(dir_path + '/../../python/openpose/Release')
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + \
            '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        sys.path.append('../../python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()
params["model_folder"] = "C:/Users/tcheo/Desktop/Git/openpose/build/models"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process Image
datum = op.Datum()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.38


def getmotion(frame, prev, x1, y1, x2, y2):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Bigger filter size, more blurry
    if (prev[0] is None):
        prev[0] = gray
    diff_frame = cv2.absdiff(gray, prev[0])
    # print(diff_frame)
    thresh_frame = cv2.threshold(diff_frame, 50, 255, cv2.THRESH_BINARY)[
        1]  # larger threshold, removing more data
    # More Iteration, more obvious the result is
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=3)
    # cv2.imshow('gray', gray)

    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    crop_img = thresh_frame[y1+10:y2+10, x1:x2]
    # print(cv2.countNonZero(thresh_frame))
    # cv2.imshow('crop', crop_img)
    # cv2.imshow('gray', gray)
    # cv2.imshow('original', frame)
    # cv2.imshow('diff', diff_frame)
    # cv2.imshow('result', thresh_frame)
    prev[0] = gray


def calc_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1)
         * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    return A, B, C


def fit_func(x, a, b, c):
    return a*(x ** 2) + b * x + c


def draww(x, y, first, highest, prev_ball, shots):
    plt.plot(x, y, 'bo')
    params = curve_fit(fit_func, x, y)
    [a, b, c] = params[0]
    x_pos1 = np.arange(first[0], prev_ball[0], 1)
    y_pos1 = []
    for i in range(len(x_pos1)):
        x_val = x_pos1[i]
        y_val = (a * (x_val ** 2)) + (b * x_val) + c
        y_pos1.append(y_val)
    if(shots == 1):
        plt.plot(x_pos1, y_pos1, linestyle='-', color='red',
                alpha=0.4, linewidth=5)  # parabola liny
    elif(shots == 2):
        plt.plot(x_pos1, y_pos1, linestyle='-', color='green',
                alpha=0.4, linewidth=5)  # parabola liny

    plt.savefig("mygraph.png")


def distance(xCoor, yCoor, prev_ball):
    return ((prev_ball[0] - xCoor) ** 2 + (prev_ball[1] - yCoor) ** 2) ** (1/2)


def detect(sess, boxes, scores, classes, num_detections, base, prev_rim, shooting, prev_ball, first, highest, x, y, xtemp, ytemp, prev_frame, frame, framewithoutbox):
    frame_expanded = np.expand_dims(frame, axis=0)
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    # for person in datum.poseKeypoints:
    #     headx, heady, conf = person[4]
        # handx, handy, handconf = person[4]
        # cv2.circle(img=frame, center=(headx, heady), radius=8,
        #            color=(0, 255, 0), thickness=-1)
        # cv2.circle(img=frame, center=(handx, handy), radius=8,
        #            color=(117, 255, 253), thickness=-1)
    headx, heady, conf = datum.poseKeypoints[0][0]
    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        # min_score_thresh=.95,
        use_normalized_coordinates=True,
        line_thickness=8)
    for i, box in enumerate(boxes[0]):
        if (scores[0][i] > 0.5):
            ymin = int((box[0] * height))
            xmin = int((box[1] * width))
            ymax = int((box[2] * height))
            xmax = int((box[3] * width))
            xCoor = int(np.mean([xmin, xmax]))
            yCoor = int(np.mean([ymin, ymax]))
            if(classes[0][i] == 2):  # Basketball
                if(ymin < (headx - 30)):  # During Shooting
                    if(not shooting[0]):
                        xtemp.clear()
                        ytemp .clear()
                        first[0] = xCoor
                        first[1] = yCoor
                        shooting[0] = True
                        shooting[1] += 1
                    if(shooting[1] == 1):
                        draw = (0, 176, 94)
                    elif(shooting[1] == 2):
                        draw = (0, 128, 255)
                    else:
                        draw = (183, 0, 255)

                    if(yCoor < highest[1]):
                        highest[0] = xCoor
                        highest[1] = yCoor

                    # print("x : ", xCoor)
                    # print("y : ", yCoor)
                    xtemp.append(xCoor)
                    ytemp.append(540 - yCoor)
                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=6,
                               color=draw, thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=6,
                               color=draw, thickness=-1)
                elif(ymin >= (headx - 30)):  # Not shooting
                    if(shooting[0] and (distance(xCoor, yCoor, prev_ball) < 100)):
                        # if(shooting[1] == 1):
                        draww(xtemp, ytemp, first, highest, prev_ball, shooting[1])
                        pts = [[xtemp[i], height - ytemp[i]] for i in range(0, len(xtemp))]
                        pts = np.array(pts, dtype=np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        if(shooting[1] == 1):
                            cv2.polylines(trace, [pts], False, (0, 0, 255), thickness=4)
                        elif(shooting[1] == 2):
                            cv2.polylines(trace, [pts], False, (0, 255, 0), thickness=4)
                        print([pts])
                        x.append(xtemp)
                        y.append(ytemp)
                        print(shooting[1], "shot")
                        print("first", first[0], first[1])
                        print("highest", highest[0], highest[1])
                        print("last", prev_ball[0], prev_ball[1])
                        print()
                        cv2.circle(img=trace, center=(first[0], first[1]),
                                   radius=10,
                                   color=(82, 168, 50), thickness=-1)
                        cv2.circle(img=trace, center=(highest[0], highest[1]),
                                   radius=10,
                                   color=(46, 240, 233), thickness=-1)
                        cv2.circle(img=trace, center=(prev_ball[0], prev_ball[1]),
                                   radius=10,
                                   color=(46, 38, 209), thickness=-1)
                        shooting[0] = False
                        shooting[2] = True

                    cv2.circle(img=frame, center=(xCoor, yCoor), radius=6,
                               color=(255, 0, 0), thickness=-1)
                    cv2.circle(img=trace, center=(xCoor, yCoor), radius=6,
                               color=(255, 0, 0), thickness=-1)

                if(shooting[2] and ymin >= prev_rim[1]):
                    # print('ymin', ymin)
                    # print('prev_rim', prev_rim[1])
                    # cv2.circle(img=trace, center=(xCoor, yCoor),
                    #            radius=20,
                    #            color=(0, 0, 255), thickness=-1)
                    shooting[2] = False

                prev_ball[0] = xCoor
                prev_ball[1] = yCoor

            if(classes[0][i] == 1):  # Rim
                # if(shooting[2] == True):
                #     getmotion(framewithoutbox, prev_frame, xmin, ymin, xmax, ymax)

                cv2.rectangle(
                    trace, (prev_rim[0], prev_rim[1]), (prev_rim[2], prev_rim[3]), (255, 255, 255), 5)
                cv2.rectangle(frame, (xmin, ymax),
                              (xmax, ymin), (0, 0, 255), 5)
                cv2.rectangle(trace, (xmin, ymax),
                              (xmax, ymin), (0, 0, 255), 5)

                # crop = frame[ymin:ymax, xmin:xmax]
                # cv2.imshow('crop', crop)
                prev_rim[0] = xmin
                prev_rim[1] = ymax
                prev_rim[2] = xmax
                prev_rim[3] = ymin
                if(ymin > base[0]):
                    base[0] = ymin

    if (shooting[2] == True):
        getmotion(framewithoutbox, prev_frame,
                  prev_rim[0], prev_rim[3], prev_rim[2], prev_rim[1])

    # print(datum.poseKeypoints[0][4])
    cv2.circle(img=frame, center=(headx, heady), radius=8,
               color=(0, 255, 0), thickness=-1)
    cv2.imshow('Pose', datum.cvOutputData)
    out.write(datum.cvOutputData)
    cv2.imwrite('trace2.png', trace)
    cv2.imshow('Tracing', trace)
    cv2.imshow('object detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()


cap = cv2.VideoCapture('video/45degree-2x.MP4')
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('45degree_testing2.AVI', fourcc,
                      fps, (int(width), int(height)))

trace = np.full((int(height), int(width), 3), 255, np.uint8)


MODEL_NAME = 'rcnn_coco_4hour'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'training/labelmap.pbtxt'
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
# Each box represents a part of the image where a particular object was detected.
boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
# Each score represent how level of confidence for each of the objects.
# Score is shown on the result image, together with the class label.
scores = detection_graph.get_tensor_by_name('detection_scores:0')
classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

prev_rim = np.array([0, 0, 0, 0])
base = np.array([0, 0])  # [0]value [1]set
# shooting or not, shots, counting basket
shooting = np.array([False, 0, False])
prev_ball = np.array([0, 0])
first = np.array([0, 0])
highest = np.array([10000, 10000])
prev_frame = np.array([None])

x = []
y = []
xtemp = []
ytemp = []

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        while True:
            ret, image_np = cap.read()
            if(ret == False):
                break
            framewithoubox = image_np.copy()
            detect(sess, boxes, scores, classes, num_detections, base, prev_rim, shooting,
                   prev_ball, first, highest, x, y, xtemp, ytemp, prev_frame, image_np, framewithoubox)
out.release()
