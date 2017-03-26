'''
    factcheck.py
'''


from nltk import word_tokenize
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


    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choiceVotes = votes.count(mode(votes))
        conf = choiceVotes/len(votes)

        return conf


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
    features = findFeatures(text)
    return voteClassifier.classify(features), voteClassifier.confidence(features)
