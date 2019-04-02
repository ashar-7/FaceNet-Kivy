import kivy
kivy.require('1.10.0')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.garden.filebrowser import FileBrowser
from kivy.utils import platform
from kivy.uix.popup import Popup
from kivy.uix.recycleview import RecycleView

from lib import *
import cv2
import os

Builder.load_file('main.kv')

tmp_path_list = None
def add_path(paths):
    global tmp_path_list
    tmp_path_list = paths

tmp_predict_path_list = ''
predicted_name = ''
def add_predict_path(path):
    global tmp_predict_path_list
    tmp_predict_path_list = path
def predict():
    global predicted_name
    net = Facenet('lib/models/nn4.small2.v1.t7', 'lib/models/shape_predictor_68_face_landmarks.dat')
    clf = FaceClassifier()
    name, prob = faces.predict(tmp_predict_path_list, net, clf)
    predicted_name = name + ', ' + str(100 * prob)[:2] + '%'

class Faces():
    def __init__(self, embeddingsPath, overwriteEmbeddingsFlag=False):
        self.persons = []
        self.embeddings = Embeddings(embeddingsPath, overwriteEmbeddingsFlag)

    def add_face(self, facenet, faceClf, name, imagePathList):
        pathDict = {name: imagePathList}
        data, example_face = faceClf.prepareDataFromImagePaths(facenet, pathDict)
        self.embeddings.save(name, data['embeddings'])

        save_path = os.path.join('saved_faces', name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        cv2.imwrite(os.path.join(save_path, name + '.png'), example_face)
        self.train(facenet, faceClf)

    def train(self, facenet, faceClf):
        import pandas as pd
        df = self.embeddings.load()
        data = {'names': df['names'], 'embeddings': pd.eval(df['embeddings'])}
        faceClf.train(data)
        faceClf.save('saved_models/svm_.dat', 'saved_models/le_.dat')

    def predict(self, im_path, net, faceClf):
        import numpy as np
        try:
            faceClf.load('saved_models/svm_.dat', 'saved_models/le_.dat')
        except:
            return None, None
        face_img = net.loadFace(im_path)
        if face_img is not None:
            face_arr = np.array(face_img)
            face_arr = np.expand_dims(face_arr, axis=0)
            E = net.forward(face_arr).reshape((1, 128))
        name, prob = faceClf.predict(E)
        return name, prob

faces = Faces('./embeddings_final.csv')

class FacesView(RecycleView):
    def __init__(self, **kwargs):
        super(FacesView, self).__init__(**kwargs)
        self.save_dir = 'saved_faces'
        self.prepare_faces()

    def prepare_faces(self):
        name_list = []

        if os.path.exists(self.save_dir):
            for name in os.listdir(self.save_dir):
                if os.path.isdir(os.path.join(self.save_dir, name)):
                    im_path = os.path.join(self.save_dir, name, name + '.png')
                    if os.path.exists(im_path):
                        name_list.append(name)

            name_data = [{'name': name} for name in name_list]
            self.data = name_data

class BrowseFilesPopup(Popup):
    def do_select(self, multiselect=True):
        homeDir = None
        if platform == 'win':
            homeDir = os.environ["HOMEPATH"]
        elif platform == 'android':
            homeDir = os.path.dirname(os.path.abspath(__file__))
        elif platform == 'linux':
            homeDir = os.environ["HOME"]
        self.fbrowser = kivy.garden.filebrowser.FileBrowser(select_string='Select',
            multiselect=multiselect, path=homeDir, filters=['*.bmp', '*.pbm', '*.pgm', '*.ppm', '*.sr', '*.ras', '*.jpeg', '*.jpg', '*.jpe', '*.jp2', '*.tiff', '*.tif', '*.png'])
        self.add_widget(self.fbrowser)
        self.fbrowser.bind(
            on_success=self._fbrowser_success,
            on_canceled=self._fbrowser_canceled,
            on_submit=self._fbrowser_success)

    def _fbrowser_success(self, fbInstance):
        if len(fbInstance.selection) == 0:
            return
        selected = []
        for file in fbInstance.selection:
            selected.append(os.path.join(fbInstance.path, file))

        self.remove_widget(self.fbrowser)
        self.fbrowser = None
        self.dismiss()

        add_path(selected)

    def _fbrowser_canceled(self, instance):
        self.fbrowser = None
        self.dismiss()

        add_path(None)

class BrowseFilesPredictPopup(Popup):
    def do_select(self, multiselect=False):
        homeDir = None
        if platform == 'win':
            homeDir = os.environ["HOMEPATH"]
        elif platform == 'android':
            homeDir = os.path.dirname(os.path.abspath(__file__))
        elif platform == 'linux':
            homeDir = os.environ["HOME"]
        self.fbrowser = kivy.garden.filebrowser.FileBrowser(select_string='Predict',
            multiselect=multiselect, path=homeDir, filters=['*.bmp', '*.pbm', '*.pgm', '*.ppm', '*.sr', '*.ras', '*.jpeg', '*.jpg', '*.jpe', '*.jp2', '*.tiff', '*.tif', '*.png'])
        self.add_widget(self.fbrowser)
        self.fbrowser.bind(
            on_success=self._fbrowser_success,
            on_canceled=self._fbrowser_canceled,
            on_submit=self._fbrowser_success)

    def _fbrowser_success(self, fbInstance):
        if len(fbInstance.selection) == 0:
            return
        for file in fbInstance.selection:
            selected = os.path.join(fbInstance.path, file)

        self.remove_widget(self.fbrowser)
        self.fbrowser = None
        self.dismiss()

        add_predict_path(selected)
        predict()

    def _fbrowser_canceled(self, instance):
        self.fbrowser = None
        self.dismiss()

        add_predict_path('')

class AddFacePopup(Popup):
    def add_face(self):
        if tmp_path_list:
            name = self.ids.face_name.text
            if name == '':
                return

            self.dismiss()
            net = Facenet('lib/models/nn4.small2.v1.t7', 'lib/models/shape_predictor_68_face_landmarks.dat')
            clf = FaceClassifier()
            faces.add_face(net, clf, name, tmp_path_list)

            add_path(None)

class TrainScreen(Screen):
    pass

class FacesScreen(Screen):
    pass

class PredictScreen(Screen):
    def set_predicted_name_and_path(self):
        self.predicted_name = predicted_name
        self.tmp_predict_path_list = tmp_predict_path_list

sm = ScreenManager()
sm.add_widget(TrainScreen(name='train'))
sm.add_widget(FacesScreen(name='faces'))
sm.add_widget(PredictScreen(name='predict'))

class FaceRecogApp(App):
    def build(self):
        self.title = 'Face Recognition App'
        return sm

if __name__ == '__main__':
    app = FaceRecogApp()
    app.run()
