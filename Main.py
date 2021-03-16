from CascadeDetector import *
'''
Main file that trains the face detection
'''

# Creates the model
def create_model():
    image_list = []
    for currentFile in glob.glob("./faces/train/face/*.pgm"):
        image_list.append((currentFile, 1))
    for currentFile in glob.glob("./faces/train/non-face/*.pgm"):
        image_list.append((currentFile, 0))
    features = make_features()
    stage = AdaBoostDetector()
    stage.train_model(image_list, features, 10)
    stage.save_model()
    return stage

# Tests the model
def test_model(model_detector):
    count = 0.0
    tot = 0.0
    for curr in glob.glob("./faces/test/faces/*.pgm"):
        im = Image.open(curr)
        np_im = np.asarray(im)
        num = model_detector.classify(np_im)
        if num == 1:
            count += 1
        tot += 1

    for curr in glob.glob("./faces/test/non-face/*.pgm"):
        im = Image.open(curr)
        np_im = np.asarray(im)
        num = model_detector.classify(np_im)
        if num == 0:
            count += 1
        tot += 1
    return count / tot


populate_integral_image()
detector = create_model()
print("Finished")
