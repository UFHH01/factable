'''
    factcheck.py
'''


from nltk import word_tokenize, sent_tokenize
import pickle
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)




documentsFile = open('saved_documents.pickle', 'rb')
documents = pickle.load(documentsFile)
documentsFile.close()


wordFeaturesFile = open('saved_word_features.pickle', 'rb')
wordFeatures = pickle.load(wordFeaturesFile)
wordFeaturesFile.close()


def findFeatures(document):
    words = word_tokenize(document)
    features = {}
    for w in wordFeatures:
        features[w] = (w in words)

    return features


featureSetsFile = open("saved_feature_set.pickle", "rb")
featureSets = pickle.load(featureSetsFile)
featureSetsFile.close()

# Training and testing set
open_file = open("mnb.pickle", "rb")
MultinomialNBClassifier = pickle.load(open_file)
open_file.close()

open_file = open("bnb.pickle", "rb")
BernoulliNBClassifier = pickle.load(open_file)
open_file.close()


open_file = open("lreg.pickle", "rb")
LogisticRegressionClassifier = pickle.load(open_file)
open_file.close()


open_file = open("lsvc.pickle", "rb")
LinearSVCClassifier = pickle.load(open_file)
open_file.close()


open_file = open("sgd.pickle", "rb")
StochasticGradientDescentClassifier = pickle.load(open_file)
open_file.close()

# Vote Classifier
voteClassifier = VoteClassifier(MultinomialNBClassifier,
                                BernoulliNBClassifier,
                                LogisticRegressionClassifier,
                                StochasticGradientDescentClassifier,
                                LinearSVCClassifier)

def factAnalysis(text):
    text = sent_tokenize(text)
    trueCount = 0.00
    falseCount = 0.00
    sentenceClass = []
    for sentence in text:
        features = findFeatures(sentence)
        if voteClassifier.classify(features):
            trueCount += 1
            sentenceClass.append(True)
        else:
            falseCount += 1
            sentenceClass.append(False)
    if trueCount > falseCount:
        return True, 1.00 - falseCount/trueCount
    else:
        return False, 1.00 - trueCount/falseCount
