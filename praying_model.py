import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.lite.python.interpreter import Interpreter

class PrayingDetector:
    def __init__(self, model_path, label_path):
        # Load model
        self.interpreter = Interpreter(model_path)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.height = self.input_details[0]['shape'][1] 
        self.width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)
        
        self.input_mean = 127.5
        self.input_std = 127.5
        
        self.min_confidence_threshold = 0.7 # Set minimun confidence threshold and label map
        with open(label_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()] # Open labels
        
    def is_bbox_too_large(self,xmax, xmin, ymax, ymin, imH, imW):
        if (xmax-xmin)*(ymax-ymin) > imH*imW*0.5:
            return True
        else:
            return False
    
    def detect_img(self, img):
        imH, imW, _ = img.shape
        body_centroid = []
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.width, self.height))
        input_data = np.expand_dims(img_resized, axis=0)
        
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
        
        # Perform object detection
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()
        
        # Retrieve detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects
        
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > self.min_confidence_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                # Check the bbox size
                if self.is_bbox_too_large(xmax,xmin,ymax,ymin,imH,imW):
                    continue

                cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

                # Draw label
                object_name = self.labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(img, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(img, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
                # Caculate body_centroid
                body_centroid.append([object_name, ((xmin+xmax)/2, (ymin+ymax)/2)])
            
        return img, body_centroid
    
if __name__ == '__main__':
    '''
    #graph = 'tflite/ssd_v2_praying20210206.tflite'
    label_path = 'tflite/prayinglabelmap.txt'
    praying_detector = PrayingDetector(graph, label_path)
    img = cv2.imread('image/new3_sitting1.jpg')
    img, body_centroid = praying_detector.detect_img(img)
    print(body_centroid)
    '''
    pass