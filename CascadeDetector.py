from AdaBoostDetector import AdaBoostDetector
from Helper import *
import pickle
import os


''' A CascadeDetector is a detection model that uses multiple
    detector layers
'''


class CascadeDetector:

    def __init__(self):
        self.stages = []            # list of detectors

    def easy_train(self, data, layer_size):
        pos_data = []
        neg_data = []
        for (x, y) in data:
            if y == 1:
                pos_data.append((x, y))
            else:
                neg_data.append((x, y))

        # Checks if features have been saved and uses it
        # If do not want to sue saved features, comment out this
        # statement and fix indent in the code in the else branch
        if os.path.exists("./features.pkl"):
            features = load_model("./features.pkl")
        else:
            features = make_features()
            for feature in features:
                feature.feature_values = get_feature_values(feature, data)
            save_features(features)
        for i in range(3):  # check if current false positive rate is
            stage = AdaBoostDetector()
            stage.easy_train(pos_data + neg_data, features, layer_size[i])
            neg_data = [(x, y) for (x, y) in neg_data if self.classify(x) == 1]
            self.stages.append(stage)


    # Trains the model
    #   data - data
    #   min_detect - minimum detection rate for a layer
    #   max_layer_fp - maximum false positive rate allowed for any layer
    #   max_fp - maximum overall false positive rate allowed
    def train_model(self, data, min_detect, max_layer_fp, max_fp):
        i = 0
        curr_f = 1.0                    # current false positive rate
        curr_d = 1.0                    # current detection rate
        pos_data = []
        neg_data = []
        for (x, y) in data:
            if y == 1:
                pos_data.append((x, y))
            else:
                neg_data.append((x, y))
        features = make_features()
        while curr_f > max_fp:          # check if current false positive rate is
            i += 1                          # higher than target
            prev_f = curr_f             # false positive rate of previous layer
            prev_d = curr_d             # detection rate of previous layer
            stage = AdaBoostDetector()
            while curr_f > max_layer_fp * prev_f:
                stage.train_model(pos_data + neg_data, features, 1)
                res = validate(pos_data, neg_data, stage)
                curr_f = res[1]
                curr_d = res[0]

                curr_h = stage.h[-1]
                while curr_d < min_detect * prev_d:
                    curr_h.thresh -= 0.05
                    res = validate(pos_data, neg_data, stage)
                    curr_d = res[0]
                    curr_f = res[1]

                if curr_f > max_fp:
                    neg_data = [(x, y) for (x, y) in neg_data if self.classify(x) == 1]
            self.stages.append(stage)

    # Classifies the given data
    # Returns 1 if positive, 0 otherwise
    #   x - data
    def classify(self, x):
        for stage in self.stages:
            predicted_y = stage.classify(x)
            if predicted_y == 0:
                return 0
        return 1

    def save_model(self):
        with open("cascade_model.pkl", "wb") as f:
            pickle.dump(self, f)
