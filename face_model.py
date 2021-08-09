import face_recognition
import pandas as pd
import cv2
import numpy as np

class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_name = []
    
    def load_all(self):
        df = pd.read_csv('face_data/known_face_db.csv')
        img_mapping = pd.Series(df.name.values, index=df.id).to_dict()
        for key in img_mapping: 
            self.add_face_data(img_path='face_data/img/'+str(key)+'.jpg' ,name=img_mapping[key])
    
    
    def add_face_data(self, img_path, name):
        new_image = face_recognition.load_image_file(img_path)
        new_face_encoding = face_recognition.face_encodings(new_image)[0]
        self.known_face_encodings.append(new_face_encoding)
        self.known_face_name.append(name)
    
    def detect_img(self, img):
        face_locations = face_recognition.face_locations(img)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        bbox_coordinate, text = [], ''
        
        #print(face_locations, face_encodings)
        if face_locations == []:
            return [], []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            best_distance = np.min(face_distances)
            if matches[best_match_index]:
                name = self.known_face_name[best_match_index]
                confidence = round(1 - best_distance, 2)
                bbox_coordinate = [(left,top), (right, bottom)]
                text = name +':'+ str(confidence)
                
        return bbox_coordinate, text
    
    def get_known_face_name(self):
        return self.known_face_name

if __name__ == '__main__':
    Face_Recognitor = FaceRecognition()
    Face_Recognitor.load_all()
    print('Known Faces: {}'.format(Face_Recognitor.get_known_face_name()))
    