# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 23:01:00 2022

@author: coban
"""


import cv2
import numpy as np

img = "2"

image = cv2.imread("extras/{}.jpg".format(img))

image_width = image.shape[1]
image_height = image.shape[0]

img_blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), swapRB = True, crop = False)

labels = [
    "Oil Storages Tank"
]

colors = ["255,0,0"]
model = cv2.dnn.readNetFromDarknet("yolov4_airbus.cfg", "yolov4_airbus_last.weights")
layers = model.getLayerNames()
OutputLayers = [ layers[layer[0] - 1] for layer in model.getUnconnectedOutLayers() ]
print(OutputLayers)
model.setInput(img_blob)

detection_layers = model.forward(OutputLayers)
print(detection_layers[0][0])
ids_list = []
boxes_list = []
confidence_list =[]

for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores =  object_detection[5:]
        predicted_id = np.argmax(scores)
        confidence = scores[predicted_id]
        
        if confidence > 0.10:
            label = labels[predicted_id]
            bounding_box = object_detection[0:4] * np.array([image_width,image_height,image_width,image_height])
            (box_center_x, box_center_y, box_witdh, box_height) = bounding_box.astype("int")
            
            start_x = int(box_center_x - (box_witdh / 2))
            start_y = int(box_center_y - (box_height / 2))
            
            ids_list.append(predicted_id)
            confidence_list.append(float(confidence))    
            boxes_list.append([start_x, start_y, int(box_witdh), int(box_height)])
            

max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5, 0.4)

box_color = (0,255,255)
label_color = (0,0,255)

for max_id in max_ids:
    max_class_id = max_id[0]
    box = boxes_list[max_class_id]
                
    start_x = box[0]
    start_y = box[1]
    box_width = box[2]
    box_height = box[3]
                
    predicted_id = ids_list[max_class_id]
    label = labels[predicted_id]
    confidence = confidence_list[max_class_id]
                
    end_x = int(start_x + box_width)
    end_y = int(start_y + box_height)
            
    label = "{}".format(np.round(confidence * 100, 2))
    print("predicted object {}".format(label))
            
    cv2.rectangle(image, (start_x,start_y), (end_x, end_y), box_color, 2)
    cv2.putText(image, label, (start_x,start_y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

image_resized = cv2.resize(image, dsize=(608,608), interpolation=cv2.INTER_CUBIC)
cv2.imshow("detection window", image_resized)
cv2.imwrite("predictions/{}.jpg".format(img),image)
