# This program preprocesses and reformats files of reviews to be passed into deception classifier 



from collections import defaultdict

import os
import json
import unicodedata

def read_folder(path, output_file, polarity):
	for i in range(1,6): 
		p = path+str(i)
		for filename in os.listdir(p):
			text = ""
			with open(p+"/"+filename) as f:
				for line in f:
			 		text += line 

			text = text.replace('\n', "")
			# ex. n_t_hilton_1 ...\n
			output_file.write(polarity+"_"+filename[:-4]+" "+text+"\n")

if __name__=='__main__':
	true_reviews = open('true_reviews.txt', 'w')
	# negative truthful
	path = os.getcwd()+"/deception_data/negative_polarity/truthful_from_Web/fold"
	read_folder(path, true_reviews, "n")
	# positive truthful 
	path = os.getcwd()+"/deception_data/positive_polarity/truthful_from_TripAdvisor/fold"
	read_folder(path, true_reviews, "p")
	true_reviews.close() 

	false_reviews = open('false_reviews.txt', 'w')
	# negative deceptive 
	path = os.getcwd()+"/deception_data/negative_polarity/deceptive_from_MTurk/fold"
	read_folder(path, false_reviews, "n")
	# positive deceptive
	path = os.getcwd()+"/deception_data/positive_polarity/deceptive_from_MTurk/fold"
	read_folder(path, false_reviews, "p")
	false_reviews.close() 

	test_reviews = open('test_reviews.txt', 'w')
	with open("filtered_noise_reviews.json") as reviews:
		for l in reviews:
			line_data = json.loads(l)
			text = line_data["text"]
			text = text.encode('ascii', 'ignore').decode('ascii')
			#print "Before: "
			#print text
			#unicodedata.normalize('NFKD', text).encode('ascii','ignore')
			text = text.replace('\n', "")
			#print "After: "
			#print text 
			test_reviews.write(str(line_data["review_id"])+" "+text+"\n") 

	test_reviews.close()
















