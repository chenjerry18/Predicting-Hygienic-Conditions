# -*- coding: utf-8 -*- 
from __future__ import division

import math
import os
import re
import random
import nltk
import operator


from collections import defaultdict


# Global class labels.
POS_LABEL = 'pos'
NEG_LABEL = 'neg'

# Path to dataset
PATH_TO_DATA = os.path.join(os.getcwd(),"review_dataset/")# FILL IN THE ABSOLUTE PATH TO THE DATASET HERE
TRAIN_DIR = os.path.join(PATH_TO_DATA, "train")
TEST_DIR = os.path.join(PATH_TO_DATA, "test")

#TEST_DIR = os.path.join(PATH_TO_DATA, "test")



def tokenize_doc(doc):
    """

    Tokenize a document and return its bag-of-words representation.
    doc - a string representing a document.
    returns a dictionary mapping each word to the number of times it appears in doc.
    """
    bow = defaultdict(float)
    tokens = doc.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        #print(token)
        bow[token] += 1.0
    return bow

def better_Tokenizer(doc):
    bow = defaultdict(float)
    txt2 = re.sub(r',|\.|\?|<br|/>|\:|;|\(|\)',"",doc)
    txt3 = re.sub(r'\'ll'," will",txt2)
    txt4 = re.sub(r'\'m'," am",txt3)
    txt5 = re.sub(r'\'s'," is",txt4)

    tokens = txt5.split()
    lowered_tokens = map(lambda t: t.lower(), tokens)
    for token in lowered_tokens:
        bow[token]+=1.0
    return bow

def nltk_tokenizer(doc):
    from nltk.tokenize import TweetTokenizer
    from nltk.stem.lancaster import LancasterStemmer
    st = LancasterStemmer()
    bow = defaultdict(float)
    #text = 'Python is a very good language. And it\'s easy to use, too'
    tknzr = TweetTokenizer()

    # sens = nltk.sent_tokenize(text)
    # print(sens)
    tokens = tknzr.tokenize(doc)
    lowered_tokens = map(lambda t: t.lower(), tokens)
    # for sent in sens:
    #     words.append(nltk.word_tokenize(sent))
    for token in lowered_tokens:
        #stemmedToken = st.stem(token)
        bow[token]+=1.0
    return bow

class NaiveBayes:
    """A Naive Bayes model for text classification."""

    def __init__(self):
        # Vocabulary is a set that stores every word seen in the training data
        self.vocab = set()
        self.posVocab = set()
        self.negVocab = set()

        # class_total_doc_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of documents in the trainning set of that class
        self.class_total_doc_counts = { POS_LABEL: 0.0,
                                        NEG_LABEL: 0.0}

        # class_total_word_counts is a dictionary that maps a class (i.e., pos/neg) to
        # the number of words in the training set in documents of that class
        self.class_total_word_counts = { POS_LABEL: 0.0,
                                         NEG_LABEL: 0.0}

        # class_word_counts is a dictionary of dictionaries. It maps a class (i.e.,
        # pos/neg) to a dictionary of word counts. For example:
        #    self.class_word_counts[POS_LABEL]['awesome']
        # stores the number of times the word 'awesome' appears in documents
        # of the positive class in the training documents.
        self.class_word_counts = { POS_LABEL: defaultdict(float),
                                   NEG_LABEL: defaultdict(float)}


    def train_model(self, num_docs=100):
        """
        This function processes the entire training set using the global PATH
        variable above.  It makes use of the tokenize_doc and update_model
        functions you will implement.

        num_docs: set this to e.g. 10 to train on only 10 docs from each category.
        """

        if num_docs is not None:
            print ("Limiting to only %s docs per clas" % num_docs)

        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        #test_path = os.path.join(TEST_DIR, "test")
        print ("Starting training with paths %s and %s" % (pos_path, neg_path))
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
        # for(p, label) in [(test_path, "test")]:
            filenames = os.listdir(p)
            allFiles = []
            for i in filenames:
                if os.path.splitext(i)[1] == '.txt':
                    allFiles.append(i)
                    # print (i)
            # print(allFiles)
            if num_docs is not None: allFiles = allFiles
            for f in allFiles:
                with open(os.path.join(p,f),'r', encoding='latin1') as doc:
                    content = doc.read()
                    #print('content',content)
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()

    def new_train_model(self, mergeSet, label):
        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        if label == POS_LABEL:
            p = pos_path
        else:
            p = neg_path
        trainset = mergeSet[0]
        for f in trainset:
                with open(os.path.join(p,f),'r', encoding='latin1') as doc:
                    content = doc.read()
                    #print('content',content)
                    self.tokenize_and_update_model(content, label)
        self.report_statistics_after_training()


    def report_statistics_after_training(self):
        """
        Report a number of statistics after training.
        """

        print ("REPORTING CORPUS STATISTICS")
        print ("NUMBER OF DOCUMENTS IN POSITIVE CLASS:", self.class_total_doc_counts[POS_LABEL])
        print ("NUMBER OF DOCUMENTS IN NEGATIVE CLASS:", self.class_total_doc_counts[NEG_LABEL])
        print ("NUMBER OF TOKENS IN POSITIVE CLASS:", self.class_total_word_counts[POS_LABEL])
        print ("NUMBER OF TOKENS IN NEGATIVE CLASS:", self.class_total_word_counts[NEG_LABEL])
        #print ("NUMBER OF DOCUMENTS IN TEST CLASS:", self.class_total_doc_counts["test"])
        #print ("NUMBER OF TOKENS IN TEST CLASS:", self.class_total_word_counts["test"])
        print ("VOCABULARY SIZE: NUMBER OF UNIQUE WORDTYPES IN TRAINING CORPUS:", len(self.vocab))
        print ("VOCABULARY SIZE: NUMBER OF POS WORDTYPES IN TRAINING CORPUS:", len(self.posVocab))
        print ("VOCABULARY SIZE: NUMBER OF NEG WORDTYPES IN TRAINING CORPUS:", len(self.negVocab))

    def update_model(self, bow, label):
        """
        IMPLEMENT ME!

        Update internal statistics given a document represented as a bag-of-words
        bow - a map from words to their counts
        label - the class of the document whose bag-of-words representation was input
        This function doesn't return anything but should update a number of internal
        statistics. Specifically, it updates:
          - the internal map the counts, per class, how many times each word was
            seen (self.class_word_counts)
          - the number of words seen for each class (self.class_total_word_counts)
          - the vocabulary seen so far (self.vocab)
          - the number of documents seen of each class (self.class_total_doc_counts)
        """
        self.class_total_doc_counts[label] += 1.0
        for k in bow:
            self.vocab.add(k)
            if label == POS_LABEL:
                self.posVocab.add(k)
            elif label == NEG_LABEL:
                self.negVocab.add(k)
            self.class_total_word_counts[label] += 1.0
            self.class_word_counts[label][k] += 1.0
            #print(k,bow[k])
        pass


    def tokenize_and_update_model(self, doc, label):
        """
        Tokenizes a document doc and updates internal count statistics.
        doc - a string representing a document.
        label - the sentiment of the document (either postive or negative)
        stop_word - a boolean flag indicating whether to stop word or not

        Make sure when tokenizing to lower case all of the tokens!
        """

        #bow = tokenize_doc(doc)
        bow = tokenize_doc(doc)
        #bow = better_Tokenizer(doc)
        #bow = nltk_tokenizer(doc)
        self.update_model(bow, label)

    def top_n(self, label, n):
        """

        Returns the most frequent n tokens for documents with class 'label'.
        """
        print('sorted: ')
        #print(self.class_word_counts[label].items())
        #print(sorted(self.class_word_counts[label].items(), key=lambda wc: -c)[:n])
        return sorted(self.class_word_counts[label].items(), key=lambda wc: (-wc[1],wc[0]))[:n]

    def p_word_given_label(self, word, label):
        """
        Implement me!

        Returns the probability of word given label (i.e., P(word|label))
        according to this NB model.
        """
        #labelDocNo = self.class_total_doc_counts[label]
        #docSum = self.class_total_doc_counts[POS_LABEL]+self.class_total_doc_counts[NEG_LABEL]
        #plabel = labelDocNo / docSum
        wFreq = self.class_word_counts[label][word]
        wSum = self.class_total_word_counts[label]
        #print("wSum",wSum)
        #pWord = wFreq / wSum

        pResult = wFreq / wSum

        return pResult

    def p_word_given_label_and_psuedocount(self, word, label, alpha):
        """
        Implement me!

        Returns the probability of word given label wrt psuedo counts.
        alpha - psuedocount parameter
        """
        #labelDocNo = self.class_total_doc_counts[label]
        #docSum = self.class_total_doc_counts[POS_LABEL]+self.class_total_doc_counts[NEG_LABEL]
        #plabel = labelDocNo / docSum
        #wordSetSize = len(self.vocab)
        #wlabelFreq = self.class_total_word_counts
        wFreq = self.class_word_counts[label][word]
        wSum = self.class_total_word_counts[label]
        #print("wSum",wSum)
        #pWord = wFreq / wSum
        if label == POS_LABEL:
            labelWordSize = len(self.posVocab)
        elif label == NEG_LABEL:
            labelWordSize = len(self.negVocab)
        #labelWordSize = len(self.class_word_counts[label])
        pResult = (wFreq+alpha) / (wSum+alpha*labelWordSize)
        #print('alpha:', alpha, 'label', label)
        #print('p_word_given_label:', pResult, 'label', label)
        #print('w_freq:', wFreq, 'label', label)
        #print('wSum:', wSum, 'label', label)
        return pResult

    def log_likelihood(self, bow, label, alpha):
        """
        Computes the log likelihood of a set of words give a label and psuedocount.
        bow - a bag of words (i.e., a tokenized document)
        label - either the positive or negative label
        alpha - float; psuedocount parameter
        """ 
        logLikelihood = 0
        for i in bow:
            logLikelihood += math.log(self.p_word_given_label_and_psuedocount(i,label,alpha))
        #print('log likelihood:', logLikelihood, 'label', label)
        return logLikelihood

    def log_prior(self, label):
        """
        Implement me!

        Returns a float representing the fraction of training documents
        that are of class 'label'.
        """
        docSum = self.class_total_doc_counts[POS_LABEL]+self.class_total_doc_counts[NEG_LABEL]
        labelDoc = self.class_total_doc_counts[label]
        logPrior = math.log(labelDoc/docSum)
        #print('log prior:', logPrior, 'label', label)
        return logPrior

    def unnormalized_log_posterior(self, bow, label, alpha):
        """
        Implement me!
        alpha - psuedocount parameter
        bow - a bag of words (i.e., a tokenized document)
        Computes the unnormalized log posterior (of doc being of class 'label').
        """
        result = self.log_prior(label)+self.log_likelihood(bow,label,alpha)
        return result

    def classify(self, bow, alpha):
        """
        Implement me!
        alpha - psuedocount parameter.
        bow - a bag of words (i.e., a tokenized document)
        Compares the unnormalized log posterior for doc for both the positive
        and negative classes and returns the either POS_LABEL or NEG_LABEL
        (depending on which resulted in the higher unnormalized log posterior).
        """        
        posScore = self.unnormalized_log_posterior(bow,POS_LABEL,alpha)
        negScore = self.unnormalized_log_posterior(bow,NEG_LABEL,alpha)
        # for i in bow:
        #    print('token', ":", bow[i])
        #print("posScore: ",posScore)
        #print("negScore: ",negScore)
        labelResult = ''
        if posScore>negScore:
            labelResult = POS_LABEL
        else:
            labelResult = NEG_LABEL
        #print("labelResult: ",labelResult)
        return labelResult

    def likelihood_ratio(self, word, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        pos_likelihood = self.p_word_given_label_and_psuedocount(word,POS_LABEL,alpha)
        neg_likelihood = self.p_word_given_label_and_psuedocount(word,NEG_LABEL,alpha)
        lr = pos_likelihood/neg_likelihood
        return lr

    def likelihood_ratio_neg(self, word, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        Returns the ratio of P(word|pos) to P(word|neg).
        """
        pos_likelihood = self.p_word_given_label_and_psuedocount(word,POS_LABEL,alpha)
        neg_likelihood = self.p_word_given_label_and_psuedocount(word,NEG_LABEL,alpha)
        lr = neg_likelihood/pos_likelihood
        return lr

    def evaluate_classifier_accuracy(self, alpha):
        """
        Implement me!

        alpha - psuedocount parameter.
        This function should go through the test data, classify each instance and
        compute the accuracy of the classifier (the fraction of classifications
        the classifier gets right.
        """

        num_doc = 20
        cor = 0
        wro = 0
        docTotal = 0

        pos_path = os.path.join(TEST_DIR,POS_LABEL)
        neg_path = os.path.join(TEST_DIR,NEG_LABEL)
        #test_path = os.path.join(TEST_DIR,"test")
        posDoc = 0
        negDoc = 0
        print ("Starting training with paths %s and %s" % (pos_path, neg_path))
        for (p, label) in [ (pos_path, POS_LABEL), (neg_path, NEG_LABEL) ]:
        #for(p,label) in [(test_path,NEG_LABEL)]:
            filenames = os.listdir(p)
            allFiles = []
            for i in filenames:
                if os.path.splitext(i)[1] == '.txt':
                    allFiles.append(i)
            if num_doc is not None: allFiles = allFiles
            for f in allFiles:
                with open(os.path.join(p,f),'r',encoding='latin1') as doc:
                    docTotal+=1
                    content = doc.read()
                    #content = content.decode('cp1252').encode('utf-8')
                    #print('token',content)
                    #self.tokenize_and_update_model(content, "test")
                    tokenizedDoc = tokenize_doc(content)
                    #tokenizedDoc = better_Tokenizer(content)
                    #tokenizedDoc = nltk_tokenizer(content)
                    #self.report_statistics_after_training()
                    #print('token',tokenizedDoc)
                    #self.update_model(tokenize_doc, label)
                    classifyResult = self.classify(tokenizedDoc, alpha)
                    #print("classifyResult: ",classifyResult)
                    #print("label: ",label)
                    if classifyResult == POS_LABEL:
                        posDoc+=1
                    if classifyResult == NEG_LABEL:
                        negDoc+=1
                    if classifyResult == label:
                        cor+=1
                    else:
                        wro+=1
                        #print("wrong document",f,"original label: ", label, "classified label: ", classifyResult)
        print("correct: ",cor)
        print("wrong",wro)
        print("classified pos doc: ",posDoc)
        print("classified neg doc: ",negDoc)
        #print("wrong",wro)
        #plot_psuedocount_vs_accuracy(alpha,cor/docTotal)
        return cor/docTotal

    def generate_trainset_devtset(self, label):
        pos_path = os.path.join(TRAIN_DIR,POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR,NEG_LABEL)
        print ("Starting trainingset with paths %s and %s" % (pos_path, neg_path))
        if label == POS_LABEL:
            p = pos_path
        else:
            p = neg_path
        filenames = os.listdir(p)
        allFiles = []
        for i in filenames:
                if os.path.splitext(i)[1] == '.txt':
                    allFiles.append(i)
        fileNum = len(allFiles)
        #fileNum = float(fileNum)
        trainset = random.sample(allFiles, int(fileNum*0.9))
        devtset = set(allFiles).difference(set(trainset))
        merge = [trainset,devtset]
        return merge

    def new_evaluate_accuracy(self, devtsetpos, devtsetneg, alpha):
        pos_path = os.path.join(TRAIN_DIR, POS_LABEL)
        neg_path = os.path.join(TRAIN_DIR, NEG_LABEL)
        posDoc = 0
        negDoc = 0
        cor = 0
        wro = 0
        docTotal = 0
        for f in devtsetpos:
            with open(os.path.join(pos_path,f),'r',encoding='latin1') as doc:
                docTotal+=1
                content = doc.read()
                tokenizedDoc = tokenize_doc(content)
                #tokenizedDoc = better_Tokenizer(content)
                #tokenizedDoc = nltk_tokenizer(content)
                classifyResult = self.classify(tokenizedDoc, alpha)
                if classifyResult == POS_LABEL:
                    posDoc+=1
                if classifyResult == NEG_LABEL:
                    negDoc+=1
                if classifyResult == POS_LABEL:
                    cor+=1
                else:
                    wro+=1  
        for f in devtsetneg:
            with open(os.path.join(neg_path,f),'r',encoding='latin1') as doc:
                docTotal+=1
                content = doc.read()
                tokenizedDoc = tokenize_doc(content)
                #tokenizedDoc = nltk_tokenizer(content)
                #tokenizedDoc = better_Tokenizer(content)
                classifyResult = self.classify(tokenizedDoc, alpha)
                if classifyResult == POS_LABEL:
                    posDoc+=1
                if classifyResult == NEG_LABEL:
                    negDoc+=1
                if classifyResult == NEG_LABEL:
                    cor+=1
                else:
                    wro+=1 
        print("correct: ",cor)
        print("wrong",wro)
        print("classified pos doc: ",posDoc)
        print("classified neg doc: ",negDoc) 
        return cor/docTotal

def produce_hw1_results():
    # PRELIMINARIES

    # QUESTION 1.1
    # uncomment the next two lines when ready to answer question 1.2
    print ('')
    print ("VOCABULARY SIZE: " + str(len(nb.vocab)))
    print ('')

    lrdict = defaultdict(float)

    # QUESTION 1.2
    # uncomment the next set of lines when ready to answer qeuestion 1.2
    print ("TOP 10 WORDS FOR CLASS " + POS_LABEL + " :")
    for tok, count in nb.top_n(NEG_LABEL, 30000):
        # print ('', tok, count)
        # print('',nb.likelihood_ratio(tok,0.01))
        sss = nb.likelihood_ratio(tok,0.01)
        lrdict[tok]=sss

        # print ('')

    sorted_x = sorted(lrdict.items(), key=operator.itemgetter(1), reverse=False)
    for tok, count in sorted_x:
        print('',tok,count)
        print('')


    # print ("TOP 10 WORDS FOR CLASS " + "test" + " :")
    # for tok, count in nb.top_n("test", 10):
    #     print ('', tok, count)
    #     print ('')

    print ("TOP 10 WORDS FOR CLASS " + NEG_LABEL + " :")
    for tok, count in nb.top_n(NEG_LABEL, 10):
        print ('', tok, count)
        # print('',nb.likelihood_ratio(tok,0.01))
        print ('')
    print ('[done.]')

    # QUESTION 2.2
    print("fantastic ",POS_LABEL," ", nb.p_word_given_label("fantastic",POS_LABEL))
    print("fantastic ",NEG_LABEL," ", nb.p_word_given_label("fantastic",NEG_LABEL))
    print("boring ",POS_LABEL," ", nb.p_word_given_label("boring",POS_LABEL))
    print("boring ",NEG_LABEL," ", nb.p_word_given_label("boring",NEG_LABEL))

    #QUESITON 5.1
    accuracy = nb.evaluate_classifier_accuracy(1.0)
    print('accuracy: ',accuracy)

    #QUESTION 5.2
    # valRange = range(1,20,1)
    # accuracies = []
    # for count in valRange:
    #     acc = nb.evaluate_classifier_accuracy(count/10)
    #     print("classify accuracy: ",acc)
    #     accuracies.append(acc)
    # plot_psuedocount_vs_accuracy(valRange,accuracies)

    #QUESTION 6.3
    print("likelihood_ratio of boring",nb.likelihood_ratio('boring',0.01))
    print("likelihood_ratio of fantastic",nb.likelihood_ratio('fantastic',0.01))
    print("likelihood_ratio of the",nb.likelihood_ratio('the',0.01))
    print("likelihood_ratio of this",nb.likelihood_ratio('this',0.01))
    print("likelihood_ratio of that",nb.likelihood_ratio('that',0.01))
    print("likelihood_ratio of it",nb.likelihood_ratio('it',0.01))

    #QUESTION 7.1
    # posFiles = nb.generate_trainset_devtset(POS_LABEL)
    # negFiles = nb.generate_trainset_devtset(NEG_LABEL)
    # nb.new_train_model(posFiles, POS_LABEL)
    # nb.new_train_model(negFiles, NEG_LABEL)
    # devtsetpos = posFiles[1]
    # devtsetneg = negFiles[1]
    # valRange = range(1,20,1)
    # accuracies = []
    # for count in valRange:
    #     acc = nb.new_evaluate_accuracy(devtsetpos,devtsetneg,count/10)
    #     print('psuedocounts: ', count/10, 'accuracy: ', acc)
    #     print('-------------------------------------------------')



def plot_psuedocount_vs_accuracy(psuedocounts, accuracies):
    """
    A function to plot psuedocounts vs. accuries. You may want to modify this function
    to enhance your plot.
    """

    import matplotlib.pyplot as plt
    
    #%matplotlib inline
    valNew = []
    #print("psuedocounts",psuedocounts)
    for i in psuedocounts:
        i = i/10
        valNew.append(i)
    plt.plot(valNew, accuracies, 'ro')
    print("psuedocounts: ", psuedocounts, "accuraices", accuracies)
    plt.xlabel('Psuedocount Parameter')
    plt.ylabel('Accuracy (%)')
    plt.title('Psuedocount Parameter vs. Accuracy Experiment')
    plt.show()

if __name__ == '__main__':
    nb = NaiveBayes()
    nb.train_model()
    #nb.train_model(num_docs=100)


    # accuracy = nb.evaluate_classifier_accuracy(1.0)
    # print('accuracy: ',accuracy)
    # # #
    produce_hw1_results()
    # valRange = range(1,20,1)
    # accuracies = []
    # for count in valRange:
    #     acc = nb.evaluate_classifier_accuracy(count/10)
    #     print("classify accuracy: ",acc)
    #     accuracies.append(acc)
    # plot_psuedocount_vs_accuracy(valRange,accuracies)
    # print("likelihood_ratio of boring",nb.likelihood_ratio('boring',0.01))
    # print("likelihood_ratio of fantastic",nb.likelihood_ratio('fantastic',0.01))
    # print("likelihood_ratio of the",nb.likelihood_ratio('the',0.01))
    # print("likelihood_ratio of this",nb.likelihood_ratio('this',0.01))
    # print("likelihood_ratio of that",nb.likelihood_ratio('that',0.01))
    # print("likelihood_ratio of it",nb.likelihood_ratio('it',0.01))


    