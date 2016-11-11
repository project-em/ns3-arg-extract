from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import pos_tag
from nltk import word_tokenize
import numpy as np
from gensim.models import Word2Vec
"""
If you use the VADER sentiment analysis tools, please cite:

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""
lexicon_file = "./arg-extract/data/vader_sentiment_lexicon.txt"
word2vec_file = "./arg-extract/data/GoogleNews-vectors-negative300.bin"

class NlpFeatures:

	def __init__(self):
		self.word2vec = Word2Vec.load_word2vec_format(word2vec_file, binary=True)

	# word2vec similarity between the topic and the nouns of the candidate sentence
	def nouns_sim(self, sentences, tagged_sentences, topics, tagged_topics):
		sim_scores = np.zeros((len(sentences)))
		for i, tagged_sentence in enumerate(tagged_sentences):
			topic = topics[i]
			topic_nouns = {word for word, pos in tagged_topics[topic] 
			if pos.startswith("NN") and word in self.word2vec.vocab}
			topic_nouns.remove("house")
			sentence_nouns = {word for word, pos in tagged_sentence 
			if pos.startswith("NN") and word in self.word2vec.vocab}
			if len(sentence_nouns) == 0:
				sim_scores[i] = 0
				continue
			similarity = self.word2vec.n_similarity(topic_nouns, sentence_nouns)
			print "topic nouns are: ", topic_nouns
			print "sentence nouns are: ", sentence_nouns
			print "similarity is: ", similarity
			sim_scores[i] = similarity
		return sim_scores

	# Sentiment Intensity based on the NLTK Vader Sentiment Intensity Analyzer
	def sentiment_scores(self, sentences):
		sent_analyzer = SentimentIntensityAnalyzer(lexicon_file=lexicon_file)
		sent_scores = np.zeros((len(sentences)))
		for i, sentence in enumerate(sentences):
			ss = sent_analyzer.polarity_scores(sentence)
			# Take the absolute value because what we care about is the overall polarity
			sent_scores[i] = abs(ss['compound'])
			print "sentence is: ", sentence
			print "sentiment_score is: ", sent_scores[i]
		return sent_scores

	def featurize(self, topics, sentences):
		tagged_sentences = [pos_tag(word_tokenize(sentence)) for sentence in sentences]
		tagged_topics = {topic:pos_tag(word_tokenize(topic)) for topic in topics}

		num_features = 2
		featurized = np.zeros((len(sentences), num_features))
		featurized[:,0] = self.nouns_sim(sentences, tagged_sentences, topics, tagged_topics)
		featurized[:,1] = self.sentiment_scores(sentences)
		return featurized

	#TODO: Cosine similarity between the topic and semantic expansions of the candidate sentence

	#TODO: part of speech or other sentence structure

	#TODO: discourse relationship features from Penn Discourse Treebank

	#TODO: pattern matching feature