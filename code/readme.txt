-------------READ ME------------------
code with comments.ipynb is the file Jiannan created to collect his code and write comments to each cell. His original code is original code1.ipynb and code2.ipynb. 

naive bayes.py is for the baseline model: naive bayes classification.

add_num_decep_to_instances.py: calculates and adds the number of labeled deceptive reviews in each instance of inspection period

calculate_coefficients.py: calculates the Spearman's correlation coefficients between demerits/penalty score and statistics of reviews

calculate_data_stats: calculates some basic statistics of reviews — word count, sentence count, document count

deception_detection.py: deception classifier

filter_deception.py: filters out deceptive reviews and generates a new file of reviews — filtered_deception_reviews.json

filter_noise.py: filters out noise reviews and generates a new file of reviews filtered_noise_reviews.json

make_deception_data.py: preprocesses and reformats files of reviews to be passed into deception classifier

