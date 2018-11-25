from sklearn.metrics import confusion_matrix

class Evaluation:
    def __init__(self, predictions, actual):
        self.confusionmatrix = self.getConfusionMatrix(predictions, actual)

        self.TP = self.confusionmatrix[0][0]
        self.TN = self.confusionmatrix[1][1]
        self.FP = self.confusionmatrix[1][0]
        self.FN = self.confusionmatrix[0][1]

    def getConfusionMatrix(self, predictions, actual):
        #print(confusion_matrix(actual, predictions, labels=[
         #     "ok", "peace", "rockon", "shaka", "thumbsup"]))
        return confusion_matrix(actual, predictions, labels=["ok", "peace", "rockon", "shaka", "thumbsup"])

    def getAccuracy(self):
        return (self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN)

    def getPrecision(self):
        return self.TP / (self.TP + self.FP)

    def getRecall(self):
        return self.TP / (self.TP + self.FN)

    def getF1(self):
        precision = self.getPrecision()
        recall = self.getPrecision()
        return (2 * precision * recall) / (precision + recall)
