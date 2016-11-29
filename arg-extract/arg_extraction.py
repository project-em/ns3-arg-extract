import numpy as np
from sklearn.linear_model import LogisticRegression
from nlp_features import NlpFeatures

class ArgumentExtractionModel:

	def __init__(self, nlp_features):
		self.clf = LogisticRegression()
		self.nlp_features = nlp_features

	# y[i] = 1 if sentences[i] is a claim on topic[i], and 0 otherwise
	def fit(self, topics, sentences, y):
		features = self.nlp_features.featurize(topics, sentences)
		print "features shape:", features.shape
		print "y length is: ", len(y)
		self.clf.fit(features, y)

	# predicts whether each sentence is likely to be a claim about the topic
	def predict(self, topics, sentences):
		features = self.nlp_features.featurize(topics, sentences)
		return self.clf.predict(features)

	# Returns the n most likely sentences as predicted by the model
	def n_most_likely(self, topics, sentences, n):
		if (n > len(sentences)):
			n = len(sentences)
		features = self.nlp_features.featurize(topics, sentences)
		# get the probability of the class being 1 for each sentence
		probabilities = self.clf.predict_proba(features)[:,1]
		tuples = zip(probabilities, np.arange(len(sentences)))
		sort_probs = np.array(tuples, 
			dtype=[('prob', 'float64'), ('index', 'int32')])
		# Sort and flip so highest probabilities come first
		sort_probs.sort(order='prob')
		sort_probs = sort_probs[::-1]
		return sort_probs[:n]['index']
