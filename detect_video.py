import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from praying_model import PrayingDetector
from face_model import FaceRecognition
from api import check_islamic_calendar


parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Input Video Path', required=True)
parser.add_argument('--output', help='Output Video Path', default='output.avi')
args = parser.parse_args()
VIDEO_PATH = args.video
OUTPUT_PATH = args.output
 
# Load models
praying_detector = PrayingDetector('tflite/ssd_v2_praying20210206.tflite', 'tflite/prayinglabelmap.txt')
Face_Recognitor = FaceRecognition()
Face_Recognitor.load_all()

def detect_video(video_path, output_path):
    # Open video file
    video = cv2.VideoCapture(video_path)
    imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output Video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (imW, imH))

    praying_time = 0
    non_praying_time = 0
    pre_body_centroid = 0
    # Check Praying Time
    check_praying_time = check_islamic_calendar()

    while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            print('Reach the end of the video')
            break

        face_coordinates, names = Face_Recognitor.detect_img(frame)

        if check_praying_time:
            frame, body_centroid = praying_detector.detect_img(frame)
            # Check Body Centroid X
            if body_centroid != []:
                if abs(pre_body_centroid - body_centroid[0][1][0]) > 5 and praying_time>=1:
                    praying_time -= 3
                else:
                    non_praying_time += 1
                pre_body_centroid = body_centroid[0][1][0]

            # Calculate Praying Time
            if body_centroid != [] and praying_time < 180:
                praying_time += 1
            elif body_centroid == [] and praying_time >= 1:
                praying_time -= 3
            cv2.putText(frame, 'Praying Time:{}s'.format(int(praying_time/30)), 
                            (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4) # Draw label text

            # Calculate non-Praying Time
            if body_centroid == [] and non_praying_time < 180:
                non_praying_time += 1
            elif body_centroid != [] and non_praying_time >= 1:
                non_praying_time -= 1
            cv2.putText(frame, 'Non-praying Time:{}s'.format(int(non_praying_time/30)), 
                            (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4) # Draw label text

        # Show Face result
        if face_coordinates != []:
            cv2.rectangle(frame, face_coordinates[0], face_coordinates[1], (0, 0, 255), 2)
            left = face_coordinates[0][0]
            bottom = face_coordinates[1][1]
            cv2.putText(frame, names, (left + 6, bottom + 10), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 0), 2)

        # Show Status
        if non_praying_time >= 180*0.66 and praying_time < 20 and 'Kevin' in names:
            cv2.putText(frame, 'Status: Time to Eat Medicine',(10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        elif not check_praying_time and 'Kevin' in names:
            cv2.putText(frame, 'Status: Time to Eat Medicine',(10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
        else:
            cv2.putText(frame, 'Status: Wait',(10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        cv2.putText(frame, 'Check Praying Time:'+str(check_praying_time), (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4) 


        out.write(frame)
        cv2.imshow('Object detector', frame) # Comment this, when you don't want to open a window

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    # Clean up
    video.release()
    out.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    detect_video(VIDEO_PATH, OUTPUT_PATH)

