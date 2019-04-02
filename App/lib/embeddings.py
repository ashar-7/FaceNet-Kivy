import os
import imghdr
import cv2
import pandas as pd

def getPathsFromDir(dir):
    if not os.path.isdir(dir):
        return None

    paths = []
    contents = os.listdir(dir)
    for f in contents:
        file_path = os.path.join(dir, f)
        if imghdr.what(file_path):
            paths.append(file_path)
    return paths

class Embeddings():
    def __init__(self, path, createNew=False):
        if createNew:
            self.createNewFile()

        fpath, ext = os.path.splitext(path)
        csv_path = fpath + '.csv' # to make sure the file is csv
        self.path = csv_path

    def createNewFile(self):
        df = pd.DataFrame(columns=['names', 'embeddings'])
        df.to_csv(self.path, index=False)
        return df

    def save(self, name, embeddings, overwrite=False):
        if not overwrite:
            df = self.load()
        else:
            df = self.createNewFile()

        for E in embeddings:
            E = list(E)
            df_new = pd.DataFrame([[name, E]], columns=['names', 'embeddings'])
            df = df.append(df_new, ignore_index=True)
        df.to_csv(self.path, index=False)

    def load(self):
        try:
            df = pd.read_csv(self.path)
        except FileNotFoundError:
            df = self.createNewFile()
        return df
