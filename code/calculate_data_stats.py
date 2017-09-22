# This program calculates some basic statistics of reviews: 
# word count, sentence count, document count 


from __future__ import division
from collections import defaultdict

import json

if __name__=='__main__':
	doc_count = 0 
	sentence_count = 0
	word_count = 0
	with open("filtered_deception_reviews.json") as reviews_set:
		for line in reviews_set:
			doc_count += 1
			review = json.loads(line)
			sents = review["text"].split('.')
			words = review["text"].split()
			sentence_count += len(sents)
			word_count += len(words) 

	print "Doc count: ", doc_count
	print "Sentences count: ", sentence_count
	print "Word count: ", word_count