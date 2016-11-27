import numpy as np
from nlp_features import NlpFeatures
from arg_extraction import ArgumentExtractionModel
from article_utils import split_sentences, read_data_file
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, precision_score
from data_prep import make_topic_map
import itertools

lexicon_filepath = "./data/vader_sentiment_lexicon.txt"
word2vec_filepath = "./data/GoogleNews-vectors-negative300.bin"

def main():
	data_prefix = "data/"
	topic_files = make_topic_map("./data/selected_topics.txt")
	num_topics = len(topic_files)

	test_topic = "This house believes that wind power should be a primary focus of future energy supply"
	
	train_sentences = []
	train_topics = []
	train_y = []
	print "loading files"
	for topic in topic_files:
		if topic != test_topic:
			sentences, y = read_data_file(topic_files[topic])
			topics = list(itertools.repeat(topic, len(sentences)))
			train_sentences.extend(sentences)
			train_topics.extend(topics)
			train_y.extend(y)

	test_file = topic_files[test_topic]
	test_sentences, test_y = read_data_file(test_file)
	test_topics = list(itertools.repeat(test_topic, len(test_sentences)))

	print "loading word2vec"
	nlp_features = NlpFeatures(word2vec_filepath, lexicon_filepath)
	arg_model = ArgumentExtractionModel(nlp_features)
	print "fitting model"
	arg_model.fit(train_topics, train_sentences, train_y)

	test_guesses = arg_model.n_most_likely(test_topics, test_sentences, 50)
	test_preds = np.zeros(len(test_y))
	for pred_index in test_guesses:
		test_preds[pred_index] = 1
	test_precision = precision_score(test_y, test_preds)
	test_recall = recall_score(test_y, test_preds)
	print "Testing Precision With Top 50 Sentences Is: ", test_precision
	print "Testing Recall With Top 50 Sentences Is: ", test_recall

if __name__ == '__main__':
	main()