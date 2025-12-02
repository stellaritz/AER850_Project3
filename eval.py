# -*- coding: utf-8 -*-
"""
#AER850
Project 3

Alina Saleem
501129840

step 3
"""

from ultralytics import YOLO
import os

#paths
#data root has the folder containing the evaluation images
DATA_ROOT = r"C:\Users\alina\Documents\GitHub\AER850_Project3\AER850_Project3\Project 3 Data"

#images the model will evaluate after the training is complete
#note: these were NOT used during training
eval_images = [
    os.path.join(DATA_ROOT, "data", "evaluation", "ardmega.jpg"),  
    os.path.join(DATA_ROOT, "data", "evaluation", "arduno.jpg"),  
    os.path.join(DATA_ROOT, "data", "evaluation", "rasppi.jpg"),  
    ]


#loading best weights

#path has the model parameters with the lowest validation loss
best_weights_path = r"C:\Users\alina\Documents\GitHub\AER850_Project3\AER850_Project3\pcb_yolo_cpu\train_run4\weights\best.pt"

print("\nUsing trained weights:")
print("  ", best_weights_path, " | exists:", os.path.exists(best_weights_path))

#failing early if weights file is not found
if not os.path.exists(best_weights_path):
    raise FileNotFoundError("best.pt not found")

#the model is loaded at the trained checkpoint
best_model = YOLO(best_weights_path)


#running predictions on 3 eval images
pred_results = best_model.predict(
    #images to run
    source=eval_images,
    #same resolution as training
    imgsz=896,
    #confidence threshold for predictions
    conf=0.25,
    #running on cpu
    device="cpu",
    save=True,
    project="pcb_yolo_cpu_eval",
    name="eval_run_from_train_run4",
)

print("Evaluation complete.")

#printing predictions details
for img_result in pred_results:
    print("\nImage:", img_result.path)
    for box in img_result.boxes:
        #predicting class index
        cls_id = int(box.cls[0])  
        #model confidence score
        conf   = float(box.conf[0]) 
        #bounding box coordinates
        xyxy   = box.xyxy[0].tolist()    
        print(f"  Class {cls_id}, conf={conf:.2f}, box={xyxy}")
        