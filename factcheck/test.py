"""
    test.py
"""

import nltk
import random
from nltk import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from nltk.classify import ClassifierI
from statistics import mode


DEBUG = True


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


# Open text dumps
trueText = open('true_text_dump.txt', 'r').read()
falseText = open('false_text_dump.txt', 'r').read()


# Extract out sentence tokens from each text dump
trueSentences = []
falseSentences = []

for l in trueText.split('\n'):
    sentences = sent_tokenize(l)

    for s in sentences:
        trueSentences.append(s)

for l in falseText.split('\n'):
    sentences = sent_tokenize(l)

    for s in sentences:
        falseSentences.append(s)

if DEBUG:
    print(trueSentences)
    print(falseSentences)


# Create documents and word features, save to pickle
documents = []
wordFeatures = []
allowedWordTypes = ['ADJ', 'ADV', 'VERB', 'N', 'CONJ', 'NUM', 'DET']

for s in trueSentences:
    documents.append((s, True))
    words = word_tokenize(s)
    pos = nltk.pos_tag(words)

    for w in pos:
        if w[1][0] in allowedWordTypes:
            wordFeatures.append(w[0].lower())

for s in falseSentences:
    documents.append((s, False))
    words = word_tokenize(s)
    pos = nltk.pos_tag(words)

    for w in pos:
        if w[1][0] in allowedWordTypes:
            wordFeatures.append(w[0].lower())


wordFeatures = nltk.FreqDist(wordFeatures)
wordFeatures = list(wordFeatures.keys())[:500]

if DEBUG:
    print(wordFeatures)

savedDocuments = open('saved_documents.pickle', 'wb')
pickle.dump(documents, savedDocuments)
savedDocuments.close()

savedWordFeatures = open('saved_word_features.pickle', 'wb')
pickle.dump(wordFeatures, savedWordFeatures)
savedWordFeatures.close()


def findFeatures(document):
    words = word_tokenize(document)
    features = {}

    for w in wordFeatures:
        features[w] = (w in words)

    return features


featureSets = []

for (s, c) in documents:
    featureSets.append((findFeatures(s), c))

savedFeatureSets = open('saved_feature_set.pickle', 'wb')
pickle.dump(featureSets, savedFeatureSets)
savedFeatureSets.close()

random.shuffle(featureSets)
print(len(featureSets))


# Training and Testing Set
trainingSet = featureSets[:350]
testingSet = featureSets[350:]


# Multinomial Naive Bayes Classifier
MultinomialNBClassifier = SklearnClassifier(MultinomialNB())
MultinomialNBClassifier.train(trainingSet)
print('Multinomial Naive Bayes Classifier Algorithm Accuracy %:', (nltk.classify.accuracy(MultinomialNBClassifier, testingSet)) * 100)

saveClassifier = open('multinomialNaiveBayes.pickle', 'wb')
pickle.dump(MultinomialNBClassifier, saveClassifier)
saveClassifier.close()

# Bernoulli Naive Bayes Classifier
BernoulliNBClassifier = SklearnClassifier(BernoulliNB())
BernoulliNBClassifier.train(trainingSet)
print('Bernoulli Naive Bayes Classifier Algorithm Accuracy %:', (nltk.classify.accuracy(BernoulliNBClassifier, testingSet)) * 100)

saveClassifier = open('bernoulliNaiveBayes.pickle', 'wb')
pickle.dump(BernoulliNBClassifier, saveClassifier)
saveClassifier.close()

# Logistic Regression Classifier
LogisticRegressionclassifier = SklearnClassifier(LogisticRegression())
LogisticRegressionclassifier.train(trainingSet)
print('Logistic Regression Classifier Algorithm Accuracy %:', (nltk.classify.accuracy(LogisticRegressionclassifier, testingSet)) * 100)

saveClassifier = open('logisticRegression.pickle', 'wb')
pickle.dump(LogisticRegressionclassifier, saveClassifier)
saveClassifier.close()

# Stochastic Gradient Descent Classifier
StochasticGradientDescentClassifier = SklearnClassifier(SGDClassifier())
StochasticGradientDescentClassifier.train(trainingSet)
print('Stochastic Gradient Descent Classifier Algorithm Accuracy %:', (nltk.classify.accuracy(StochasticGradientDescentClassifier, testingSet)) * 100)

saveClassifier = open('stochasticGradientDescent.pickle', 'wb')
pickle.dump(StochasticGradientDescentClassifier, saveClassifier)
saveClassifier.close()

# LinearSVC Classifier
LinearSVCClassifier = SklearnClassifier(LinearSVC())
LinearSVCClassifier.train(trainingSet)
print('LinearSVC Classifier Algorithm Accuracy %:', (nltk.classify.accuracy(LinearSVCClassifier, testingSet)) * 100)

saveClassifier = open('linearSVC.pickle', 'wb')
pickle.dump(LinearSVCClassifier, saveClassifier)
saveClassifier.close()

# Voted Classifier
votedClassifier = VoteClassifier(MultinomialNBClassifier,
                                 BernoulliNBClassifier,
                                 LogisticRegressionclassifier,
                                 StochasticGradientDescentClassifier,
                                 LinearSVCClassifier)

def factAnalysis(text):
    features = findFeatures(text)
    return votedClassifier.classify(features), votedClassifier.confidence(features)

