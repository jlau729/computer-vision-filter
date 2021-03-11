from AdaBoostDetector import AdaBoostDetector
from Helper import *


''' A CascadeDetector is a detection model that uses multiple
    detector layers
'''


class CascadeDetector:

    def __init__(self):
        self.stages = []            # list of detectors

    # Trains the model
    #   pos_data - data that should return 1
    #   neg_data - data that should return 0
    #   min_detect - minimum detection rate for a layer
    #   max_layer_fp - maximum false positive rate allowed for any layer
    #   max_fp - maximum overall false positive rate allowed
    def train_model(self, pos_data, min_detect, max_layer_fp, max_fp):
        i = 0
        neg_data = set()
        curr_f = 1.0                    # current false positive rate
        curr_d = 1.0                    # current detection rate
        tot_data = list(pos_data)
        features = make_features()
        while curr_f > max_fp:          # check if current false positive rate is
            i += 1                          # higher than target
            prev_f = curr_f             # false positive rate of previous layer
            prev_d = curr_d             # detection rate of previous layer
            stage = AdaBoostDetector()
            while curr_f > max_layer_fp * prev_f:
                stage.train_model(tot_data, features, 1)
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
                    for (x, y) in tot_data:
                        predicted_y = stage.classify(x)
                        if predicted_y == 1 and y == 0:
                            neg_data.add((x, y))
                            if (x, y) in pos_data:
                                pos_data.remove((x, y))
                        else:
                            if (x, y) not in pos_data:
                                pos_data.append((x, y))
                            if (x, y) in neg_data:
                                neg_data.remove((x, y))
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
