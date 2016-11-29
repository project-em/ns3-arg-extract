import numpy as np
from collections import defaultdict
from article_utils import split_sentences
import re

def make_topic_map(topic_filename, topic_prefix, write=False):
	if write:
		mode = "w"
	else:
		mode = "r"
	topic_files = {}
	topics_file = open(topic_filename, "r")
	topics = topics_file.read().splitlines()
	for line in topics[1:]:
		split = line.split("\t")
		topic = split[0]
		file_name = split[1].strip()
		print "file name is:", file_name
		topic_files[topic] = open(topic_prefix + file_name + ".data", mode)
	return topic_files

def prep_articles():
	topic_files = make_topic_map("CE-EMNLP-2015.v3/topics.txt", "data/", write=True)

	# store the claims by topic
	claims_file = open("CE-EMNLP-2015.v3/claims.txt", "r")
	claims = claims_file.read().splitlines()
	claims_by_topic = defaultdict(set)
	for line in claims[1:]:
		split = line.split("\t")
		topic = split[0]
		# the second entry is the unmodified text from the wikipedia sentence
		if topic in topic_files:
			claims_by_topic[topic].add(split[2])
	claims_file.close()

	articles_file = open("CE-EMNLP-2015.v3/articles.txt", "r")
	articles_by_topic = articles_file.read().splitlines()
	articles_file.close()

	topic_count = defaultdict(int)
	# skip header line
	# restrict range for testing
	for line in articles_by_topic:
		split = line.split("\t")
		topic = split[0]
		if topic in topic_files:
			topic_file = topic_files[topic]
			article_filenum = split[2]
			article_file = open("CE-EMNLP-2015.v3/articles/clean_" + article_filenum + ".txt", "r")
			article = article_file.read()
			article = article.replace(" [REF]", "")
			article = article.replace("[REF]", "")
			
			article_file.close()
			# replace newlines within sentence because newline will be the delimeter in the file
			sentences = split_sentences(article)
			for sentence in sentences:
				# go through set of claims and see if any are contained in this sentence
				match = None
				for claim in claims_by_topic[topic]:
					if claim in sentence:
						match = claim
				if not(match is None):
					claims_by_topic[topic].remove(match)
					topic_file.write(str(topic_count[topic]) + "\t" + sentence + "\t" + str(1) + "\n")
				else:
					topic_file.write(str(topic_count[topic]) + "\t" + sentence + "\t" + str(0) + "\n")
				topic_count[topic] += 1

	for topic, file in topic_files.items():
		file.close()

if __name__ == '__main__':
	prep_articles()
	