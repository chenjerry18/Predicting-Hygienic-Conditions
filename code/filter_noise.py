# This program filters out noise reviews and generates a new file of reviews
# filtered_noise_reviews.json


from collections import defaultdict

import json

if __name__=='__main__':
	output = open('filtered_noise_reviews.json', 'w')

	id_reviews = defaultdict(list) 
	with open("vegas_reviews.json") as reviews:
		for l in reviews:
			line_data = json.loads(l) 
			# review = line_data["text"]
			business_id = line_data["business_id"] 
			if (business_id not in id_reviews):
				id_reviews[business_id] = [line_data] 
			else: 
				id_reviews[business_id].append(line_data)  

	avg_rates = defaultdict(float) 
	for business_id, reviews in id_reviews.items():
		count = len(reviews)
		total = 0
		for r in reviews:
			total += r["stars"] 
		avg = total/count 
		avg_rates[business_id] = avg 

	for business_id, reviews in id_reviews.items():
		avg = avg_rates[business_id]
		for r in reviews:
			if (abs(avg-r["stars"]) < 2):
				json_data = json.dumps(r, output)
				output.write(json_data+'\n') 

	output.close() 