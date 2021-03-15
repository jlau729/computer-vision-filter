import cv2
from PIL import Image
import glob

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Censors the faces in the given image with the given sticker
# original-image: string path to the image we want to censor the faces in
# sticker_path: image we want to use to censor all the faces
# returns a new image that is the result of the original image censored by the sticker
def censor_face(original_image, sticker_path):
    img = cv2.imread(original_image)
    layered_copy = Image.open(original_image)
    sticker = Image.open(sticker_path) # sticker we want to use to censor
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        sticker = sticker.resize((w, h))
        layered_copy.paste(sticker, (x, y))
    return layered_copy

censor_face("selfie.jpg", "../pokeball.png").save("censored_selfie.jpg", format = "png")
censor_face("paris.jpg", "../pokeball.png").save("censored_paris.jpg", format = "png")