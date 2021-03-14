from Helper import *
from sklearn.feature_selection import SelectPercentile, f_classif
import pickle

WIDTH = 19
HEIGHT = 19

''' An AdaBoostDetector represents a detection model that uses
    an AdaBoost approach
'''


class AdaBoostDetector:

    def __init__(self):
        self.h = []              # list of weak classifiers

        # Trains the model with the given training data of images

    def easy_train(self, data, features, num_rounds):
        sample_w = initialize_weights(data)
        for i in range(num_rounds):
            normalize(sample_w)
            curr_models = []

            # Get matrix of feature to feature values along with expected y values
            feature_m, y = make_feature_m(features, data)

            # Select features more likely to be helpful
            indices = SelectPercentile(f_classif, percentile=10).fit(feature_m.T, y)\
                .get_support(indices=True)

            # Change features to only contain the helpful features corresponding to
            # the indices array
            #features = features[indices]
            features_updated = np.array(features)[indices]
            feature_m = feature_m[indices]

            # Assign feature values to the respective feature and sort them
            get_feature_values(features_updated, feature_m)
            for feature in features_updated:
                model = make_model(feature, sample_w, y)
                curr_models.append(model)
            if len(curr_models) > 0:
                min_h = curr_models[0]
                for h in curr_models:
                    if h.e < min_h.e:
                        min_h = h
                adjust_weights(min_h, sample_w, data)
                self.h.append(min_h)

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

    def save_model(self):
        with open("model.pkl", "wb") as f:
            pickle.dump(self, f)
