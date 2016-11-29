import numpy as np
from nlp_features import NlpFeatures
from arg_extraction import ArgumentExtractionModel
from article_utils import split_sentences, read_data_file
from sklearn.metrics import roc_curve, auc, recall_score, accuracy_score, precision_score
from data_prep import make_topic_map
from av_precision import average_precision
import itertools

lexicon_filepath = "./data/vader_sentiment_lexicon.txt"
word2vec_filepath = "./data/GoogleNews-vectors-negative300.bin"

def main():
	data_prefix = "data/"
	# The file selected_topics contains 10 topics to use for 10-fold cross validation.
	topic_files = make_topic_map("./data/selected_topics.txt", "./data/")
	num_topics = len(topic_files)

	print "loading files"
	sentence_map = {topic: read_data_file(topic_file) for topic, topic_file in topic_files.items()}

	print "loading word2vec"
	nlp_features = NlpFeatures(word2vec_filepath, lexicon_filepath)
	arg_model = ArgumentExtractionModel(nlp_features)

	av_precs = np.zeros(len(topic_files))

	for i, test_topic in enumerate(topic_files):
		train_sentences = []
		train_topics = []
		train_y = []
		for topic in sentence_map:
			if topic != test_topic:
				sentences, y = sentence_map[topic]
				topics = list(itertools.repeat(topic, len(sentences)))
				train_sentences.extend(sentences)
				train_topics.extend(topics)
				train_y.extend(y)

		test_sentences, test_y = sentence_map[test_topic]

		test_correct = set(np.nonzero(test_y)[0].flat)
		test_topics = list(itertools.repeat(test_topic, len(test_sentences)))
		
		print "fitting model for test topic ", test_topic
		print "test correct indices: ", test_correct
		arg_model.fit(train_topics, train_sentences, train_y)

		test_guesses = arg_model.n_most_likely(test_topics, test_sentences, 50)
		av_precision = average_precision(test_correct, test_guesses, 50)
		av_precs[i] = av_precision
		print "Average Precision for topic ", test_topic, " is: ", av_precision
	
	print "Mean average precision is: ", np.mean(av_precs)

if __name__ == '__main__':
	main()