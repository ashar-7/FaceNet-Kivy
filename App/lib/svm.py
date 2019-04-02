from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cv2
import os
import pickle

from .facenet import Facenet
from .embeddings import Embeddings, getPathsFromDir

class FaceClassifier():
    def __init__(self, C=1.0):
        self.svm = SVC(C, kernel='linear', probability=True)
        self.le = LabelEncoder()

    def train(self, data):
        labels = self.le.fit_transform(data['names'])
        self.svm.fit(data['embeddings'], labels)

    def save(self, svm_path, le_path):
        with open(svm_path, 'wb') as f:
            f.write(pickle.dumps(self.svm))

        with open(le_path, 'wb') as f:
            f.write(pickle.dumps(self.le))

    def load(self, svm_path, le_path):
        with open(svm_path, 'rb') as f:
            self.svm = pickle.loads(f.read())

        with open(le_path, 'rb') as f:
            self.le = pickle.loads(f.read())

    def predict(self, embeddings):
        preds = self.svm.predict_proba(embeddings)[0]
        i = np.argmax(preds)
        prob = preds[i]
        name = self.le.classes_[i]
        return name, prob

    def prepareDataFromImagePaths(self, facenet, pathDict):
        names = []
        embeddings = []
        example_face = None # to display in face view
        for name, paths in pathDict.items():
            for p in paths:
                face_img = facenet.loadFace(p)
                if face_img is not None:
                    face_arr = np.array(face_img)
                    face_arr = np.expand_dims(face_arr, axis=0)
                    E = facenet.forward(face_arr).squeeze()

                    embeddings.append(E)
                    names.append(name)
                    example_face = facenet.loadFace(p, False) if example_face is None else example_face

        data = {'names': names, 'embeddings': embeddings}
        return data, example_face

    def prepareDataFromDir(self, facenet, train_dir):
        names = []
        embeddings = []

        for dir in os.listdir(train_dir):
            paths = getPathsFromDir(os.path.join(train_dir, dir))
            for p in paths:
                face_img = facenet.loadFace(p)
                if face_img is not None:
                    face_arr = np.array(face_img)
                    face_arr = np.expand_dims(face_arr, axis=0)
                    E = facenet.forward(face_arr).squeeze()

                    embeddings.append(E)
                    names.append(dir)

        data = {'names': names, 'embeddings': embeddings}
        return data
