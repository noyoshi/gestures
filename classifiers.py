import collections
import pickle
import pandas as pd

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from evaluations import Evaluation

class Classifiers(object):
    def __init__(self):
        # Models
        self.svm = svm.SVC()
        self.knn = KNeighborsClassifier(n_neighbors=6)
        self.cart = DecisionTreeClassifier()
        self.models = [("svm", self.svm), ("knn", self.knn), ("cart", self.cart)]
        
        # Training / evaluating data
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None

        # Flags
        self.models_trained = False 
        self.data_loaded = False
    
    def load_training_data(self, train_file="training_data.txt", test_file="testing_data.txt"):
        """Opens the training and testing data"""
        df_training = pd.read_csv(train_file)
        self.train_data, self.train_label = df_training.loc[:, df_training.columns != 'label'], df_training['label']

        df_testing = pd.read_csv(test_file)
        self.test_data, self.test_label = df_testing.loc[:,df_testing.columns != 'label'], df_testing['label']
        self.data_loaded = True
    
    def train_models(self):
        """Trains all the models"""
        if not self.data_loaded: 
            self.load_training_data()

        for name, model in self.models:
            # Train and save the model
            model.fit(self.train_data, self.train_label)
            pickle.dump(model, open("models/{}".format(name), "wb"))   

        self.models_trained = True
    
    def evaluate_models(self):
        """Evaluates the models"""
        self.check_init() 
        for name, model in self.models:
            predictions = model.predict(self.test_data)
            eval_data = Evaluation(predictions, self.test_label)   
            f1 = eval_data.getF1()
            print("-"*40)
            print("\tModel: {}".format(name))
            print("\tPredictions: ")
            print("\t" + ', '.join(predictions))
            print("\tTrue Gestures: ")
            print("\t" + ', '.join(self.test_label))
            print("\tF1: {}".format(f1))

    def check_init(self):
        """Checks to see if the data is loaded into the models etc"""
        if not self.models_trained:
            self.train_models()

        if not self.data_loaded: 
            self.load_training_data()

    def make_guess(self, df):
        """Make a prediction based on data frame"""
        self.check_init() 
        guesses = collections.Counter()
        for name, model in self.models:
            predictions = model.predict(df)
            prediction = predictions[0]
            guesses[prediction] += 1
        # Return the most common prediction TODO make this more sophisticated?
        return guesses.most_common(1)[0]


if __name__ == '__main__':
    c = Classifiers()
    c.evaluate_models()
