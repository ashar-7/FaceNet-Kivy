import cv2
import numpy as np

from .align import FaceAligner

class Facenet():
    def __init__(self, modelPath, shapePredPath, imgDim=96):
        self.imgDim = imgDim

        self.model = cv2.dnn.readNetFromTorch(modelPath)
        self.aligner = FaceAligner(shapePredPath)

    def loadFace(self, path, preprocessFlag=True):
        def preprocess(img, preprocessFlag):
            face =  self.aligner.align(self.imgDim, img)
            if face is not None:
                if not preprocessFlag:
                    return face

                face = face.astype('float32') / 255.0
                return face
            return None

        img = cv2.imread(path)
        if img is None:
            return None
        return preprocess(img, preprocessFlag)

    def forward(self, imgs):
        """
            expects BGR images in the range 0-1.
        """
        blob = cv2.dnn.blobFromImages(imgs, 1, (96, 96), swapRB=True)
        self.model.setInput(blob)
        preds = self.model.forward()
        return preds
