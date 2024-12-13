import os

import cv2
from roboflow import Roboflow
from ultralytics import YOLO


def download_dataset():
    rf = Roboflow(api_key="ulcNVi2QOZNvT0xomzed")
    project = rf.workspace("duckietown-3pgbf").project("duckiebots-traffic-image")
    version = project.version(1)
    dataset = version.download("yolov5")


def train_yolo():
    model = YOLO('model_weights/yolov8n.pt')
    root = os.getcwd()

    results = model.train(data=root+'/datasets/detection/data.yaml',
                          imgsz=640,
                          epochs=250,
                          batch=8,
                          name='yolom_model')



if __name__=="__main__":
    train_yolo()
    # path = 'test_images'
    # images = os.listdir(path)
    # count = 0
    # for image in images:
    #     fpath = path + image
    #     test_yolo(fpath, count)
    #     count += 1
    # print(images)
    # test_yolo('/home/WVU-AD/rp00052/PycharmProjects/pole_data_collection/vehicle_detectionV1.1-1/test/images/a8126_png.rf.c97428269ff595c120cf81053a14a6d2.jpg')