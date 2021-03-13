from CascadeDetector import *
'''
Main file that trains the face detection
'''
file_to_integral = {}

def populate_integral_image():
    for currentFile in glob.glob("./faces/train/face/*.pgm"):
        im = Image.open(currentFile)
        np_im = np.asarray(im)
        integral_image = make_integral(np_im)
        file_to_integral[currentFile] = integral_image
    for currentFile in glob.glob("./faces/train/non-face/*.pgm"):
        im = Image.open(currentFile)
        np_im = np.asarray(im)
        integral_image = make_integral(np_im)
        file_to_integral[currentFile] = integral_image

def create_model():
    image_list = []
    for currentFile in glob.glob("./faces/train/face/*.pgm"):
        image_list.append((currentFile, 1))
    for currentFile in glob.glob("./faces/train/non-face/*.pgm"):
        image_list.append((currentFile, 0))
    cascade_detector = CascadeDetector()
    cascade_detector.easy_train(image_list, [2, 9, 12])
    cascade_detector.save_model()
    return cascade_detector

def test_model(detector):
    count = 0.0
    tot = 0.0
    for curr in glob.glob("./faces/test/faces/*.pgm"):
        im = Image.open(curr)
        np_im = np.asarray(im)
        num = detector.classify(np_im)
        if num == 1:
            count += 1
        tot += 1

    for curr in glob.glob("./faces/test/non-face/*.pgm"):
        im = Image.open(curr)
        np_im = np.asarray(im)
        num = detector.classify(np_im)
        if num == 0:
            count += 1
        tot += 1
    return count / tot

populate_integral_image()
cascade_detector = create_model()
cascade_detector.save_model()
res = test_model(cascade_detector)
print(res)
