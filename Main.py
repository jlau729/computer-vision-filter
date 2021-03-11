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
    print(len(image_list))

create_model()