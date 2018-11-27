import collections
import pandas as pd

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from evaluations import Evaluation

class Classifiers(object):
    def __init__(self):
        # Models
        self.svm = svm.SVC()
        self.knn = KNeighborsClassifier(n_neighbors=6)
        self.cart = DecisionTreeClassifier()
        self.models = [
                ("svm", self.svm), 
                ("knn", self.knn), 
                ("cart decision tree", self.cart)]
        
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
    
    def train(self):
        """Trains all the models"""
        if not self.data_loaded: 
            self.load_training_data()
        
        for name, model in self.models:
            # Train and save the model
            model.fit(self.train_data, self.train_label)

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
        
        predictions = self.make_guess(self.test_data, True)
        eval_data = Evaluation(predictions, self.test_label)
        print("-"*40)
        print("\tModel: {}".format("Voting Ensamble Model"))
        print("\tPredictions: ")
        print("\t" + ', '.join(predictions))
        print("\tTrue Gestures: ")
        print("\t" + ', '.join(self.test_label))
        print("\tF1: {}".format(f1))
        
    def check_init(self):
        """Checks to see if the data is loaded into the models etc"""
        if not self.data_loaded: 
            self.load_training_data()

        if not self.models_trained:
            self.train()

    def make_guess(self, df, multi=False):
        """Make a prediction based on data frame"""
        self.check_init() 

        # Using the three models and doing manual voting
        guesses = collections.Counter()
        multi_guesses = []
        for name, model in self.models:
            predictions = model.predict(df)
            if multi:
                multi_guesses.append(predictions)
                continue
            prediction = predictions[0]
            guesses[prediction] += 1

        if not multi:
            return guesses.most_common(1)[0]

        preds = []
        for idx in range(len(multi_guesses[0])):
            guesses = collections.Counter()
            for row in range(len(multi_guesses)):
                guesses[multi_guesses[row][idx]] += 1
            preds.append(guesses.most_common(1)[0][0])
        return preds

if __name__ == '__main__':
    c = Classifiers()
    c.evaluate_models()
