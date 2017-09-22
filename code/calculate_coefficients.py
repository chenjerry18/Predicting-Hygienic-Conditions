# This program calculates the Spearman's correlation coefficients 
# between demerits/penalty score and statistics of reviews: 
# review count, avg review length 
# avg review ratings, negative review count
# fake review count 
#


import scipy.stats 
import os
import pylab as plt


def get_coefficients(path):
	thresholds = [10,15,20,25]
	fake_reviews_coefficients = [] 
	review_count_coefficients = [] 
	neg_review_count_coefficients = []
	avg_rating_coefficients = []
	avg_review_len_coefficients = [] 

	for t in thresholds: 
		demerits = [] 
		fake_reviews_l = []
		review_count_l = []
		neg_review_count_l = []
		avg_rating_l = []
		avg_review_len_l = []
		for filename in os.listdir(path):
			if not (filename.startswith('.')):  
				with open(path+"/"+filename,"r") as f:
					line = f.readline() 
					line_data = line.split(" ")
					fake_reviews = int(line_data[0])
					review_words = int(line_data[1])
					avg_rating = float(line_data[2])
					neg_reviews = int(line_data[3])
					review_count = int(line_data[4])
					demerit = float(line_data[5])
					if(demerit > t):
						demerits.append(demerit)
						fake_reviews_l.append(fake_reviews)
						review_count_l.append(review_count)
						neg_review_count_l.append(neg_reviews)
						avg_rating_l.append(avg_rating)
						avg_review_len_l.append(review_words/review_count) 

		spearman = scipy.stats.spearmanr(fake_reviews_l, demerits)
		fake_reviews_coefficient = spearman[0]
		spearman = scipy.stats.spearmanr(review_count_l, demerits)
		review_count_coefficient = spearman[0] 
		spearman = scipy.stats.spearmanr(neg_review_count_l, demerits)
		neg_review_count_coefficient = spearman[0] 
		spearman = scipy.stats.spearmanr(avg_rating_l, demerits)
		avg_rating_coefficient = spearman[0]
		spearman = scipy.stats.spearmanr(avg_review_len_l, demerits)
		avg_review_len_coefficient = spearman[0]

		fake_reviews_coefficients.append(fake_reviews_coefficient)
		review_count_coefficients.append(review_count_coefficient)
		neg_review_count_coefficients.append(neg_review_count_coefficient)
		avg_rating_coefficients.append(avg_rating_coefficient)
		avg_review_len_coefficients.append(avg_review_len_coefficient) 

	return fake_reviews_coefficients, review_count_coefficients, neg_review_count_coefficients, avg_rating_coefficients, avg_review_len_coefficients 

if __name__=='__main__':
	thresholds = [10,15,20,25]
	filtered_path = os.getcwd()+"/coefficients_data/filtered/posNew"
	filtered_fake_reviews_coefficients, filtered_review_count_coefficients, filtered_neg_review_count_coefficients, filtered_avg_rating_coefficients, filtered_avg_review_len_coefficients = get_coefficients(filtered_path) 
	unfiltered_path = os.getcwd()+"/coefficients_data/unfiltered/posNew"
	unfiltered_fake_reviews_coefficients, unfiltered_review_count_coefficients, unfiltered_neg_review_count_coefficients, unfiltered_avg_rating_coefficients, unfiltered_avg_review_len_coefficients = get_coefficients(unfiltered_path) 


	fig = plt.figure(1)
	plt.title("Penalty Score vs. Review Count")
	plt.xlabel("Inspection Penalty Score Threshold")
	plt.ylabel("Coefficient")
	plt.plot(thresholds, filtered_review_count_coefficients, '-o', label='filtered review count')
	plt.plot(thresholds, unfiltered_review_count_coefficients, '-o', label='unfiltered review count')
	plt.legend(loc='upper right') 



	fig = plt.figure(2)
	plt.title("Penalty Score vs. Negative Review Count")
	plt.xlabel("Inspection Penalty Score Threshold")
	plt.ylabel("Coefficient")
	plt.plot(thresholds, filtered_neg_review_count_coefficients, '-o', label='filtered negative review count')
	plt.plot(thresholds, unfiltered_neg_review_count_coefficients, '-o', label='unfiltered negative review count')
	plt.legend(loc='upper right')

	fig = plt.figure(3)
	plt.title("Penalty Score vs. Average Review Rating")
	plt.xlabel("Inspection Penalty Score Threshold")
	plt.ylabel("Coefficient")
	plt.plot(thresholds, filtered_avg_rating_coefficients, '-o', label='filtered avg review rating')
	plt.plot(thresholds, unfiltered_avg_rating_coefficients, '-o', label='unfiltered avg review rating')
	plt.legend(loc='upper right')

	fig = plt.figure(4)
	plt.title("Penalty Score vs. Average Review Length")
	plt.xlabel("Inspection Penalty Score Threshold")
	plt.ylabel("Coefficient")
	plt.plot(thresholds, filtered_avg_review_len_coefficients, '-o', label='filtered avg review length')
	plt.plot(thresholds, unfiltered_avg_review_len_coefficients, '-o', label='unfiltered avg review length')
	plt.legend(loc='upper right')

	fig = plt.figure(5)
	plt.title("Penalty Score vs. Fake Review Count")
	plt.xlabel("Inspection Penalty Score Threshold")
	plt.ylabel("Coefficient")
	plt.plot(thresholds, filtered_fake_reviews_coefficients, '-o')
	plt.plot(thresholds, unfiltered_fake_reviews_coefficients, '-o', label='unfiltered fake review count')
	plt.legend(loc='upper right')

	plt.show()












