# This program filters out deceptive reviews and generates a new file of reviews:
# filtered_deception_reviews.json


from collections import defaultdict

import json

if __name__=='__main__':
	output = open('filtered_deception_reviews.json', 'w')
	review_ids = []
	with open("outputSVM.txt") as outputs:
		for l in outputs:
			line_data = l.split()
			r_id = line_data[0]
			polarity = line_data[1]
			if (polarity == "T"):
				review_ids.append(r_id)
	
	with open("filtered_noise_reviews.json") as reviews:
		for l in reviews:
			line_data = json.loads(l)
			if (line_data["review_id"] in review_ids):
				json_data = json.dumps(line_data, output)
				output.write(json_data+'\n')

	output.close()