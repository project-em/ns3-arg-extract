import nltk
import os, logging, json, boto
import numpy as np
import itertools
from flask import Flask, request, render_template, jsonify
from flask_restplus import Api, Resource, fields
from arg_extraction import ArgumentExtractionModel
from nlp_features import NlpFeatures
from article_utils import split_sentences, read_data_file
from boto.s3.connection import S3Connection

ROOT_URL = os.getenv('ROOT_URL', 'localhost')
VERSION_NO = os.getenv('VERSION_NO', '1.0')
APP_NAME = os.getenv('APP_NAME', "Devil's Advocate Sentiment")
DEBUG = os.getenv('DEBUG', False)

AWS_ACCESS = os.getenv('AWS_ACCESS', '')
AWS_SECRET = os.getenv('AWS_SECRET', '')
AWS_BUCKET = os.getenv('AWS_BUCKET', 'ns3seniordesign')
AWS_FILE = os.getenv('AWS_FILE', 'GoogleNews-vectors-negative300.bin')

NUM_SENTENCES = 10
lexicon_filepath = "./arg-extract/data/vader_sentiment_lexicon.txt"
word2vec_filepath = "./arg-extract/data/GoogleNews-vectors-negative300.bin"
wind_power_topic = "This house believes that wind power should be a primary focus of future energy supply"
data_files_prefix = "./arg-extract/data/"
data_files = ['wind_power.data']
google_lib = 'arg-extract/data/GoogleNews-vectors-negative300.bin'

def pct_download(curr, total):
	print '{0} out of {1} downloaded.'.format(curr, total)

if not os.path.isfile(google_lib):
	conn = S3Connection(AWS_ACCESS, AWS_SECRET, host='s3.us-east-2.amazonaws.com')
	bucket = conn.get_bucket(AWS_BUCKET)
	bucket.get_key(AWS_FILE).get_contents_to_filename(google_lib, cb=pct_download, num_cb=10)

app = Flask(__name__)
nlp_features = NlpFeatures(word2vec_filepath, lexicon_filepath)
arg_model = ArgumentExtractionModel(nlp_features)

api = Api(app, version=VERSION_NO, title=APP_NAME)
public_ns = api.namespace('api/v1', description='Public methods')

article = api.model('Article', {
	'article': fields.String(description='Article body', required=True)
})

@public_ns.route('/parse')
class Parse(Resource):

	@public_ns.expect(article)
	def post(self):
		data = json.loads(request.data)
		sentences = split_sentences(data['article'].encode('ascii', 'ignore'))
		topics = list(itertools.repeat(wind_power_topic, len(sentences)))
		top_sentence_indices = arg_model.n_most_likely(topics, sentences, NUM_SENTENCES)
		top_sentences = []
		for s_index in top_sentence_indices:
			top_sentences.append(sentences[s_index])
		return top_sentences

def main():
	nltk.download('averaged_perceptron_tagger')
	nltk.download('punkt')

	sentences = []
	y = []
	for filename in data_files:
		filename = data_files_prefix + filename
		f_sentences, f_y = read_data_file(open(filename, 'r'))
		sentences.extend(f_sentences)
		y.extend(f_y)
	topics = list(itertools.repeat(wind_power_topic, len(sentences)))
	arg_model.fit(topics, sentences, y)
	app.run(host="0.0.0.0", port=5000)

if __name__ == '__main__':
	main()
	
