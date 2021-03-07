from Helper import *

WIDTH = 24
HEIGHT = 24

''' An AdaBoostDetector represents a detection model that uses
    an AdaBoost approach
'''


class AdaBoostDetector:

    def __init__(self):
        self.h = []              # list of weak classifiers

    # Trains the model with the given training data of images
    def train_model(self, data, features, num_rounds):
        sample_w = initialize_weights(data)
        for i in range(num_rounds):
            normalize(sample_w)
            curr_models = []
            for feature in features:
                feature_values = get_feature_values(feature, data)
                model = make_model(feature, sample_w, feature_values)
                curr_models.append(model)
            min_h = curr_models[0]
            for h in curr_models:
                if h.e < min_h.e:
                    min_h = h
            adjust_weights(min_h, sample_w)
            self.h.append(min_h)

    # Classifies the given image based on the strong classifier
    # Returns 1 if a face, 0 otherwise
    #   x - image
    def classify(self, x):
        if len(self.h) == 0:
            return 0
        tot_w = 0
        tot_alpha = 0
        for model in self.h:
            predicted_y = model.classify(x)
            tot_w += model.alpha * predicted_y
            tot_alpha += model.alpha
        if tot_w >= tot_alpha / 2:
            return 1
        return 0
