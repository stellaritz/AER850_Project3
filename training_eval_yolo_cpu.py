# -*- coding: utf-8 -*-
"""

AER850
Project 3

Alina Saleem
501129840

step 2 

"""
from ultralytics import YOLO   
import torch                   
import os                      

#path set up

#data root points to the folder that has all the training, validation and testing subfolder
#data yaml is the config file with each class name and image path 
DATA_ROOT = r"C:\Users\alina\Documents\GitHub\AER850_Project3\AER850_Project3\Project 3 Data"
DATA_YAML = r"C:\Users\alina\Documents\GitHub\AER850_Project3\AER850_Project3\Project 3 Data\data\data.yaml"

#not to be used for training
eval_images = [
    os.path.join(DATA_ROOT, "Evaluation", "ardmega.jpg"),  
    os.path.join(DATA_ROOT, "Evaluation", "arduino.jpg"),  
    os.path.join(DATA_ROOT, "Evaluation", "rasppi.jpg"),  
    ]

#device info
# we will be training on my personal cpu

#load  YOLOv11 NANO
model = YOLO("yolo11n.pt")

#training
results = model.train(
    #dataset configuration
    data=DATA_YAML,             
    #less than 200
    epochs=40,
    #enough to detect small parts
    imgsz=896,                  
    #small batch 
    batch=4,                   
    #force cpu
    device="cpu",               
    #single process dataloader
    workers=0,                 
    project="pcb_yolo_cpu",     
    name="train_run",           
)

print("Training complete.")

