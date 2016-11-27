import re

def split_sentences(article):
	article = article.replace("\xe2\x80\x9c", "\"")
	article = article.replace("\xe2\x80\x9d", "\"")
	article = article.replace("\xe2\x80\x98", "'")
	article = article.replace("\xe2\x80\x99", "'")
	article = article.replace("\xe2\x80\x93", "-")
	article = article.replace("\s.\s", ".\s")
	sentences = [sentence.strip().replace("\n", " ")
			for sentence in re.split("(?<=[a-z0-9)\"'])\.\s", article) 
			if not sentence.isspace()]
	return sentences

def read_data_file(datafile):
	splitlines = [line.split('\t') for line in datafile.read().splitlines()]
	sentences = [split[1].decode('utf-8', 'ignore')  for split in splitlines]
	y = [int(split[2]) for split in splitlines]
	return sentences, y