from PIL import Image
import glob
from CascadeDetector import *

'''
Main file that trains the face detection
'''

def create_model():
    image_list = []
    for currentFile in glob.glob("./faces/train/face/*.pgm"):
        im = Image.open(currentFile)
        image_list.append((im, 1))
    for currentFile in glob.glob("./faces/train/non-face/*.pgm"):
        im = Image.open(currentFile)
        image_list.append((im, 0))
    cascadeDetect = CascadeDetector()
    cascadeDetect.train_model(image_list, 50, 50, 80)
    print("done")

create_model()