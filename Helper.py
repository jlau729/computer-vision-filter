from PIL import Image
import numpy as np
import math
import glob
import pickle as pk

WIDTH = 19
HEIGHT = 19
file_to_integral = {}
'''Region of an image with (x, y) as top left corner 
'''


class Region:

    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    # Returns sum of pixel within this region based on the given integral image
    def get_sum(self, integral_image):
        return integral_image[self.x + self.w][self.y + self.h] + \
               integral_image[self.x][self.y] \
               - integral_image[self.x + self.w][self.y] - \
               integral_image[self.x][self.y + self.h]


''' A WeakLearner represents a weak classifier
'''


class WeakLearner:

    def __init__(self, feature):
        self.feature = feature              # feature
        self.thresh = 0.0                     # threshold
        self.p = 0.0                          # polarity
        self.alpha = 0.0                      # alpha
        self.e = 0.0                          # error

    # Classifies the data and returns the predicted results
    # If positive, return 1 and 0 otherwise
    #   x - data
    def classify(self, x):
        im = Image.open(x)
        np_im = np.asarray(im)
        integral_image = make_integral(np_im)
        feature_value = self.feature.apply_feature(integral_image)
        predicted_y = 0
        if self.p == 1:
            if feature_value > self.thresh:
                predicted_y = 1
        else:
            if feature_value < self.thresh:
                predicted_y = 1
        return predicted_y


''' A Feature represents a Haar-like feature
    It contains rectangles that adds to the feature value
    and rectangles that are negative and lower the feature value
'''


class Feature:

    def __init__(self, pos_rect, neg_rect):
        self.pos_rect = pos_rect                # list of rectangles that are added to sum
        self.neg_rect = neg_rect                # list of rectangles that are subtracted from sum
        self.feature_values = {}

    # Applies the feature to the given integral image
    # Returns the feature value
    def apply_feature(self, integral_image):
        feature_sum = 0
        for region in self.pos_rect:
            feature_sum += region.get_sum(integral_image)
        for region in self.neg_rect:
            feature_sum -= region.get_sum(integral_image)
        return feature_sum


def load_model(model):
    with open(model, "rb") as f:
        return pk.load(f)


def populate_integral_image():
    for currentFile in glob.glob("./faces/train/face/*.pgm"):
    #for currentFile in glob.glob("./small-faces/face/*.pgm"): # smaller dataset to test code
        im = Image.open(currentFile)
        np_im = np.asarray(im)
        integral_image = make_integral(np_im)
        file_to_integral[currentFile] = integral_image
    for currentFile in glob.glob("./faces/train/non-face/*.pgm"):
    #for currentFile in glob.glob("./small-faces/non-face/*.pgm"): # smaller dataset to test code
        im = Image.open(currentFile)
        np_im = np.asarray(im)
        integral_image = make_integral(np_im)
        file_to_integral[currentFile] = integral_image


# Makes the feature matrix where each row is one feature and each column is a feature value
# for a sample data
# Returns a tuple (feature matrix, sample y)
#   features - list of features
#   data - sample data
def make_feature_m(features, data):

    # Create matrix where each row is a feature and each column is the feature value
    feature_values = np.zeros((len(features), len(data)))

    # Array that saves expected values for each sample data
    sample_y = []
    for i in range(len(data)):
        sample_y.append(data[i][1])

    i = 0
    for feature in features:
        j = 0
        for (x, y) in data:
            feature_values[i][j] = feature.apply_feature(file_to_integral[x])
            j += 1
        i += 1
    return feature_values, sample_y


# Normalizes the given list of sample weights
#   sample_w - list of sample weights
def normalize(sample_w):
    tot = sum(sample_w)
    for i in range(len(sample_w)):
        sample_w[i] = sample_w[i] / tot


# Calculates and returns a tuple of (detection rate, false positive rate) on the given
# data set
#   pos_data - data that should return 1
#   neg_data - data that should return 0
#   detector - detector to test data on
def validate(pos_data, neg_data, detector):
    detect_count = 0.0
    for (x, y) in pos_data:
        res = detector.classify(x)
        if res == 1:
            detect_count += 1

    error_count = 0.0
    for (x, y) in neg_data:
        res = detector.classify(x)
        if res == 1:
            error_count += 1

    return detect_count / len(pos_data), error_count / len(neg_data)


# Initializes the sorted feature to feature values mapping for each feature
#   feature - list of features
#   m - matrix where number of rows is number of features and number of columns
#       is number of sample data
def get_feature_values(features, m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            temp = m[i][j]
            features[i].feature_values[j] = temp
        sorted(features[i].feature_values.items(), key=lambda kv: kv[1])


# Looks for optimal threshold and polarity
# Optimal is the minimum sum of the errors of the sample weights
#   feature - the feature to be optimized
#   sample_w - the sample weights after applying the
#               feature
def make_model(feature, sample_w, sample_y):
    ret = WeakLearner(feature)
    tot_pos = 0
    tot_neg = 0
    for i in range(len(sample_w)):
        if sample_y[i] == 1:
            tot_pos += sample_w[i]
        else:
            tot_neg += sample_w[i]

    print("Total Positive: " + str(tot_pos))
    print("Total negative: " + str(tot_neg))
    min_error = tot_pos + tot_neg
    min_sample = None
    curr_pos = 0
    curr_neg = 0
    p = 0
    for i in feature.feature_values:
        if sample_y[i] == 1:
            curr_pos += sample_w[i]
        else:
            curr_neg += sample_w[i]
            err_1 = curr_neg + (tot_pos - curr_pos)
            err_2 = curr_pos + (tot_neg - curr_neg)
            error = min(err_1, err_2)
            if error < min_error:
                min_error = error
                min_sample = i
                if error == err_1:
                    p = 1
                else:
                    p = 0
    ret.thresh = feature.feature_values[min_sample]
    ret.p = p
    print("Min error value: " + str(min_error))
    correct_rate = math.log(1.0 - min_error, 10)
    denom = math.log(min_error, 10) - correct_rate
    ret.alpha = math.log(1.0, 10) - denom
    ret.e = min_error
    return ret


# Adjusts the sample weights
#   feature - the feature that has been applied
#   sample_w - the sample weights
#   data - images
def adjust_weights(model, sample_w, data):
    for i in range(len(sample_w)):
        predicted_y = model.classify(data[i][0])
        sample_res = 1
        if predicted_y == data[i][1]:
            sample_res = 0
        sample_w[i] = sample_w[i] * math.pow((model.e * 1.0 / (1 - model.e)), 1 - sample_res)


# Initializes and returns a map of sample weight pairs
#   data - tuples of form (x, y) where x is image and y is
#           correct label
def initialize_weights(data):
    tot_pos = 0
    tot_neg = 0
    for (x, y) in data:
        if y == 1:
            tot_pos += 1
        else:
            tot_neg += 1

    w = []
    for (x, y) in data:
        if y == 1:
            w.append(1.0 / (2 * tot_pos))
        else:
            w.append(1.0 / (2 * tot_neg))
    return w


# Makes and returns a list of features
#   width - width of image
#   height - height of image
def make_features():
    features = []
    for w in range(1, WIDTH + 1):
        for h in range(1, HEIGHT + 1):
            x = 0
            while x + w < WIDTH:
                y = 0
                while y + h < HEIGHT:
                    rect = Region(x, y, w, h)
                    right_rect = Region(x + w, y, w, h)
                    bottom_rect = Region(x, y + h, w, h)
                    right_edge_rect = Region(x + (2 * w), y, w, h)
                    bottom_right = Region(x + w, y + h, w, h)

                    if x + (2 * w) < WIDTH:
                        # features.append(Feature([rect], [right_rect]))
                        features.append(Feature([right_rect], [rect]))

                    if y + (2 * h) < HEIGHT:
                        features.append(Feature([rect], [bottom_rect]))
                        # features.append(Feature([bottom_rect], [rect]))

                    if x + (3 * w) < WIDTH:
                        features.append(Feature([right_rect], [rect, right_edge_rect]))
                        # features.append(Feature([rect, right_edge_rect], [right_rect]))

                    if x + (2 * w) < WIDTH and y + (2 * h) < HEIGHT:
                        features.append(Feature([rect, bottom_right], [right_rect, bottom_rect]))
                        # features.append(Feature([right_rect, bottom_rect], [rect, bottom_right]))
                    y += 1
                x += 1
    return features


# Makes and return an integral image for the given image
#   im - image
def make_integral(im_data):
    integral_im = np.zeros(im_data.shape)

    for row in range(im_data.shape[0]):
        for col in range(im_data.shape[1]):
            curr_sum = im_data[row][col]
            if row != 0:
                curr_sum += integral_im[row - 1][col]
            if col != 0:
                curr_sum += integral_im[row][col - 1]
            if row != 0 and col != 0:
                curr_sum -= integral_im[row - 1][col - 1]
            integral_im[row][col] = curr_sum
    return integral_im
