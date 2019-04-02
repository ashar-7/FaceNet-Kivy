# FaceNet-Kivy
FaceNet Face recognition GUI using kivy

A GUI for Facenet Face Recognition using kivy. This is just a fun project I made to explore kivy (I'm disappointed with it).
I didn't package the app coz I was bored so just run it yourself.

## Models
There are 2 models, one for generating face embeddings (facenet) and the other for face alignment.
Download both the models from the following links and put them in the directory `lib/models/` with the names `nn4.small2.v1.t7` and `shape_predictor_68_face_landmarks.dat` respectively.

`nn4.small2.v1.t7`: https://cmusatyalab.github.io/openface/models-and-accuracies/ 
or direct link: https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7

`shape_predictor_68_face_landmarks.dat` (direct link): http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 

## Run
Make sure you've put the models in the correct directory and run the app:
`python app.py` (Please run it from the directory where the app.py lies).

## Screenshots
![alt text](https://github.com/ashar-7/FaceNet-Kivy/tree/master/screenshots/training.jpg)
![alt text](https://github.com/ashar-7/FaceNet-Kivy/tree/master/screenshots/faces.jpg)
![alt text](https://github.com/ashar-7/FaceNet-Kivy/tree/master/screenshots/predict.jpg)


