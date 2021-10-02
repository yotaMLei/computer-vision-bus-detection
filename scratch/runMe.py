

# TODO: Change to only relative paths!!!

import numpy as np
import ast
import os
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
# from imageio import imread
import numpy as np
import os
import sys
import timeit
import glob
from PIL import Image, ImageOps
import cv2
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import load_model

sys.path.append(os.path.abspath('../'))
from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss
# from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
# from keras_layers.keras_layer_DecodeDetections import DecodeDetections
# from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
# from keras_layers.keras_layer_L2Normalization import L2Normalization
#
# from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast
#
# from data_generator.object_detection_2d_data_generator import DataGenerator
# from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
# from data_generator.object_detection_2d_geometric_ops import Resize
# from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
RATIO = 3648 / 512  # translate bounding boxes of original images to 512X512 images
# annResultsFilename = r'annotationsResults.txt'
weights_path ='trained_models/ssd_bus_detection_model.h5'
SHOW_IMAGES = False
SAVE_IMAGES = False

img_height = 512
img_width = 512
BLACK = [0, 0, 0]

def write_result_to_annotations_file(filename, image_name, ann_result):
    ann_file = open(filename, 'a')

    if image_name[:-4] == '.JPG':
        ann_file.write(image_name + ':')
    else:
        ann_file.write(image_name[:-4] + '.JPG:')

    for j in range(len(ann_result)):
        ann_file.write(str(ann_result[j]))
        if j < len(ann_result) - 1:
            ann_file.write(',')
    ann_file.write('\n')
    ann_file.close()

def preProcessImg(img):
    # top, bottom, left, right = get_padding_size(img)
    top = 0
    bottom = int((3648 - 2736))
    left = 0
    right = 0
    dstImg = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    resized_image = cv2.resize(dstImg, (img_height, img_width))
    return resized_image

def check_colorful(cropped_img):
    h, w, c = cropped_img.shape
    cropped_img = cropped_img[2 * h // 7:5 * h // 6, w // 5:4 * w // 5, :]
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
    # print(f'hsv average:{np.average(hsv[:, :, :], axis=(0, 1))}')
    min_0 = hsv[:, :, 0].min()
    # print(f'{hsv[:, :, 0].min()},{hsv[:, :, 0].max()},{hsv[:, :, 1].min()},{hsv[:, :, 1].max()},{hsv[:, :, 2].min()},{hsv[:, :, 2].max()}')

    red1 = cv2.inRange(hsv, (0, 0.30, 50), (20, 1, 255))
    red2 = cv2.inRange(hsv, (325, 0.30, 50), (360, 1, 255))
    orange = cv2.inRange(hsv, (25, 0.25, 0), (50, 1, 255)) # change to 20 - 60 hue
    green = cv2.inRange(hsv, (60, 0.20, 0), (180, 1, 255))
    blue = cv2.inRange(hsv, (180, 0.20, 50), (275, 1, 255))
    white = cv2.inRange(hsv, (0, 0, 150), (360, 0.26, 255))
    gray = cv2.inRange(hsv, (0, 0, 50), (360, 0.26, 140))
    black = cv2.inRange(hsv, (0, 0, 0), (360, 1, 50))

    g_mask = cv2.inRange(hsv, (0, 0, 0), (360, 255, 255))
    total = np.sum(g_mask // 255)
    # print("all_sum",total)
    orange = np.sum(orange // 255) * 100 / total
    # print("orange_sum", orange)
    green = np.sum(green // 255) * 100 / total
    # print("green_sum", green)
    blue = np.sum(blue // 255) * 100 / total
    # print("blue_sum", blue)
    red = np.sum(red1 // 255) * 100 / total + np.sum(red2 // 255) * 100 / total
    # print("red_sum", red)
    white = np.sum(white // 255) * 100 / total
    # print("white_sum", white)
    gray = np.sum(gray // 255) * 100 / total
    # print("gray_sum", gray)
    black = np.sum(black // 255) * 100 / total
    # print("black_sum (ignore that)", black)

    # classes = ['background','green', 'yellow', 'white', 'gray', 'blue', 'red']
    color_conf = [0, green, orange, white, gray, blue, red]
    return color_conf, np.argmax(color_conf)

def predict_color(ssd_color, cv_color, cv_color_conf):

    # print("***** color prediction *****")
    # print(f'SSD color = {colors[ssd_color]} / {cv_color_conf[ssd_color]:.2f}, CV color = {colors[cv_color]} / {cv_color_conf[cv_color]:.2f}')

    if ssd_color == cv_color:
        # cv has strong prediction and conforms with ssd prediction
        # print(f'chose color {colors[cv_color]}')
        # print("****************************")
        return cv_color
    elif (cv_color_conf[cv_color]-cv_color_conf[ssd_color] > 15) and (ssd_color != cv_color):
        # cv has strong prediction but it's different from the ssd prediction
        # the cv prediction conf is stronger than ssd prediction conf
        # in this case take the cv prediction over the ssd
        # print(f'chose color {colors[cv_color]}')
        # print("****************************")
        return cv_color
    elif (cv_color_conf[cv_color]-cv_color_conf[ssd_color] <= 15) and (ssd_color != cv_color):
        # cv has strong prediction but it's different from the ssd prediction
        # the cv prediction conf is not much stronger than ssd prediction conf
        # in this case take the ssd prediction over the cv
        # print(f'chose color {colors[ssd_color]}')
        # print("****************************")
        return ssd_color

def run(myAnnFileName, buses):
    model = ssd_512(image_size=(img_height, img_width, 3),
                    n_classes=6,
                    mode='inference_fast',
                    l2_regularization=0.0055,
                    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                    # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 128, 256, 512],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.50,
                    iou_threshold=0.40,  # orig 0.45 - todo: 0 if is in the same color, 1 if differant color
                    top_k=1000,
                    nms_max_output_size=1000)  # orig = 400

    model.load_weights(weights_path)

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    #################################################################################
    confidence_threshold = 0.8

    for img_path in glob.iglob(buses + '/*.jp*'):
        input_images = []

        img_name = img_path.split("\\")[-1]
        orig_images = cv2.imread(img_path)
        resized_images = preProcessImg(orig_images)
        resized_images = cv2.cvtColor(resized_images, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(resized_images)  # Pillow
        img = image.img_to_array(img)  # keras preprocessing
        input_images.append(img)
        input_images = np.array(input_images)

        # orig_images = cv2.cvtColor(orig_images, cv2.COLOR_BGR2RGB)

        y_pred = model.predict(input_images)
        # print(f'predicted_shape: {y_pred.shape}')
        # print(f'predicted: {y_pred[0, :7, :]}')
        y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
        # print(f'y_pred_thresh: {y_pred_thresh}')
        np.set_printoptions(precision=2, suppress=True, linewidth=90)

        classes = ['background', 'green', 'yellow', 'white', 'gray', 'blue', 'red']
        box_set = set()
        color_set = set()
        ann_result = []
        for box in y_pred_thresh[0]:
            # print(f'box[0] is {box[0]}')
            if (box[0] in box_set):
                continue # skip this bounding box
            else:
                box_set.add(box[0])
                ssd__pred_color = int(box[0])
                ssd_conf = box[1]

            xmin = box[-4] * RATIO
            ymin = box[-3] * RATIO
            xmax = box[-2] * RATIO
            ymax = box[-1] * RATIO

            cropped_image = input_images[0, int(ymin / RATIO):int(ymax / RATIO), int(xmin / RATIO):int(xmax / RATIO), :]
            # print(f'croped shape : {input_images.shape}')
            # print(f'croped shape : {cropped_image.shape}')
            cv_color_conf, cv_pred_color = check_colorful(cropped_image)

            # if the color already been chosen, choose the 2nd strongest color (there can only be 1 bus in each color!)
            if cv_pred_color in color_set:
                cv_color_conf[cv_pred_color] = 0
                cv_pred_color = np.argmax(cv_color_conf)

            color = predict_color(ssd__pred_color, cv_pred_color, cv_color_conf)
            if (color in color_set) and (ssd_conf < 0.9):
                continue  # skip this bounding box

            # add color to set of previously chosen colors
            color_set.add(color)

            annotation = [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin), int(box[0])]
            ann_result.append(annotation)
        print(f'finished processing image {img_name}...')

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Create Project annotations file
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        write_result_to_annotations_file(myAnnFileName, img_name, ann_result)















