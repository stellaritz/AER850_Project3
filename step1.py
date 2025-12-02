# -*- coding: utf-8 -*-
"""
AER850
Project 3

Alina Saleem
501129840
"""

#main computer vision library
import cv2
import numpy as np
import matplotlib.pyplot as plt 

#loading the original image

image_path =r"C:\Users\alina\Documents\GitHub\AER850_Project3\AER850_Project3\Project 3 Data\motherboard_image.JPEG"


#reading the image from disk into numpy array
img_bgr=cv2.imread(image_path)

if img_bgr is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")
    
#convert to RGB for plotting

#swapping channels from BGR to RGB
img_rgb=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#converting to grayscale 

#the grayscale will simplify the problem by removing the 
#colour channels when the color is not critical

#turn 3 channel color image into 1 channel intensity image
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#blur to reduce noise using gaussian blur, low pass filter to
#resemble a smooth convolution kernal

gray_blur = cv2.GaussianBlur(
    #input image
    gray,        
    #kernal size or 5x5 window
    (5, 5),      
    #sigma =0 
    0            
)

#thresholding to get a binary image
#converts grayscale to black/white, subtracting the background
#from the PCB itself
#Otsu to choose thr optimal threshold from the histogram, 
_, thresh = cv2.threshold(
    #blurred grayscale image
    gray_blur,                     
    #initial threshold value
    0,                              
    #pixel value above the threshold (white)
    255,                            
    #binary threshold combined with Otsu automatic threshold
    cv2.THRESH_BINARY + cv2.THRESH_OTSU  
)



#white pcb 
#fraction of white pixels 255 in the threshold image
white_ratio = np.mean(thresh == 255)   

#if pixels > %50 are white, it is the background
if white_ratio > 0.5:                   
    #inverting black and white, pcb stays white and background is black
    thresh = cv2.bitwise_not(thresh)    
    
    
#detecting edges
#using canny edge detector to predict edges
edges = cv2.Canny(
    #input image
    gray_blur,  
    #lower hysteresis threshold
    50,          
    #upper hysteresis threshold 
    150          
)


#finding the contours
#the contours are the outlines of the connected white regions in binary image 
contours, hierarchy = cv2.findContours(
    #binary image where pcb component is white
    thresh,                
    #keeping the outermost contours
    cv2.RETR_EXTERNAL,      
    #compressing straight-line points to save memory 
    cv2.CHAIN_APPROX_SIMPLE 
)


#in the case that openCV cannot detect white blobs
if len(contours) == 0:                     
    raise RuntimeError("no contours were found, try adjusting threshold or blur parameters.")

#picking the largerst contour
#"The area of the contour can be used to filter out small contours." 

#picking the contour with the maximum area 
largest_cnt = max(contours, key=cv2.contourArea)  

#creating empty mask image 
#the mask is the same height/width as the original
#single channel, starting with complete black or 0 
mask = np.zeros(gray.shape, dtype=np.uint8)


#drawing and filling the contour on mask
#drawing largest contour in white in 255 and filling it
#pcb is white,  background is black
cv2.drawContours(
    #image itself 
    mask,                 
    #list of contours to draw
    [largest_cnt],        
    #-1 is draw all contours
    contourIdx=-1,        
    #white or 255 ison the 1-channel mask
    color=255,            
    #fills the whole shape 
    thickness=cv2.FILLED  
)



#applying mask with bitwise and
#bitwie_and will keep pixels where the mask is not zero or white
#everything else will be black
pcb_bgr = cv2.bitwise_and(
    #original image
    img_bgr,  
    #second input 
    img_bgr,  
    #masking array 
    mask=mask 
)

#converting to RGB for projection in matplotlib plots
pcb_rgb = cv2.cvtColor(pcb_bgr, cv2.COLOR_BGR2RGB)


#saving the intermediate results 
#images include:
    #grayscale image
cv2.imwrite("step1_gray.png", gray)               
#threshold binary image
cv2.imwrite("step1_thresh.png", thresh)            
#canny edges
cv2.imwrite("step1_edges.png", edges)             
#final binary mask
cv2.imwrite("step1_mask.png", mask)                
#extracted pcb
cv2.imwrite("step1_pcb_extracted.png", pcb_bgr)    



#showing all results side by side

#plotting window
plt.figure(figsize=(12, 8))   
#position 1 in 2x3 grid
plt.subplot(2, 3, 1)         
#original RGB image
plt.imshow(img_rgb)           
#title
plt.title("original RGB") 
#no axis ticks    
plt.axis("off")              

#pos. 2
plt.subplot(2, 3, 2)          
#showing grayscale
plt.imshow(gray, cmap="gray") 
plt.title("grayscale")
plt.axis("off")

#pos.3
plt.subplot(2, 3, 3)         
#threshold image
plt.imshow(thresh, cmap="gray") 
plt.title("thresholded")
plt.axis("off")

#pos.4
plt.subplot(2, 3, 4)        
#edges
plt.imshow(edges, cmap="gray")  
plt.title("edges (canny)")
plt.axis("off")

#pos.5
plt.subplot(2, 3, 5)         
#mask
plt.imshow(mask, cmap="gray") 
plt.axis("off")

#pos.6
plt.subplot(2, 3, 6)          
#extracted PCB
plt.imshow(pcb_rgb)           
plt.title("extracted PCB")
plt.axis("off")

plt.tight_layout()            
plt.show()                    
