# This program calculates and adds the number of labeled deceptive reviews 
# in each instance of inspection period




import deception_detection
import os

if __name__=='__main__':
	reviewDict = {}
	reviewLabel = []
	reviewList = []
	truereviewDict,truereviewList,truereviewLabel = deception_detection.readFile('true_reviews.txt',1, "train")
	falsereviewDict,falsereviewList,falsereviewLabel = deception_detection.readFile('false_reviews.txt',0, "train")
	reviewList = truereviewList + falsereviewList
	reviewLabel = truereviewLabel + falsereviewLabel
	#path = os.getcwd()+"/coefficients_data/filtered/posNew"
	path = os.getcwd()+"/coefficients_data/unfiltered/posNew"
	for filename in os.listdir(path):
		if not (filename.startswith('.')):
			testreviewDict,testreviewList,testreviewLabel = deception_detection.readFile(path+"/"+filename,0, "test")
			num_decep = deception_detection.Testing(reviewList,reviewLabel,testreviewList,testreviewDict)
			with open(path+"/"+filename, 'r') as original: data = original.read()
			with open(path+"/"+filename, 'w') as modified: modified.write(str(num_decep)+" "+data)




