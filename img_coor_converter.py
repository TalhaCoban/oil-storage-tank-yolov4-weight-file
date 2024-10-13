import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image

IMG_WITDH = 2560
IMG_HEIGHT = 2560

def bound_converter(x):
    try:
        bounds = []
        a = str(x).split("(")[1].split(")")[0].split(",")
        for i in a:
            bounds.append(int(str(i).strip()))
        return bounds
    except:
        return x

annotations = pd.read_csv("annotations.csv")
annotations["bounds"] = annotations["bounds"].apply(bound_converter)

print("Total number of bounding boxes is {}".format(annotations.image_id.count()))
print("Total number of images is {}".format(annotations["image_id"].unique().__len__()))
print("Number of classes is {}".format(annotations["class"].unique().__len__()))

images = annotations.image_id.unique()
bounding_box_numbers = []
for image in images:
    bounding_box_numbers.append(annotations[annotations["image_id"] == image]["bounds"].count())

def img_coor_to_yolo_coor(x):
    center_x = (x[2] + x[0]) / 2
    center_y = (x[3] + x[1]) / 2
    width = x[2] - x[0]
    height = x[3] - x[1]
    return list((np.array([center_x, center_y, width, height]).reshape([2,2]) / np.array([IMG_WITDH, IMG_HEIGHT])).reshape([4,]))

annotations["yolo_coor"] = annotations["bounds"].apply(img_coor_to_yolo_coor)

def write_labels():
    for image in images:
        text = ""
        labels = annotations[annotations["image_id"] == image]["yolo_coor"]
        for label in labels:
            text +=  "0 {} {} {} {}\n".format(str(label[0]), str(label[1]), str(label[2]), str(label[3]))
        file = open("labels/{}.txt".format(image),"w")
        file.write(text)
        file.close()

def test_train():
    train = []
    test = []
    train_file = open("airbus_data/airbus_training.txt","w")
    test_file = open("airbus_data/airbus_testing.txt","w")
    train = images[:88]
    test = images[88:]
    text = ""
    for i in train:
        text += "{}/{}.jpg\n".format("airbus_data/images",i)
    train_file.write(text)
    text = ""
    for j in test:
        text += "{}/{}.jpg\n".format("airbus_data/images",j)
    test_file.write(text)
    train_file.close()
    test_file.close()

print(annotations.head())


