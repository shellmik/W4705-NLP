import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    # add START, END and sequence length (not sure)
    sequence_copy = sequence.copy()
    sequence_copy.insert(0, 'START')
    sequence_copy.insert(len(sequence_copy), 'STOP')
    for i in range(0, n-2):
        sequence_copy.insert(0, 'START')

    list_ngrams = []
    length = len(sequence_copy)
    for i in range(length - n + 1):
        gram = tuple()
        for k in range(n):
            gram += (sequence_copy[i + k],)
        list_ngrams.append(gram)

    return list_ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)   #
        self.lexicon = get_lexicon(generator)  # SET of words that appear in txt more than once
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)  # less frequent words are replaced by UNK.
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {}
        self.wordcount = 0
        self.sentencecount = 0

        ##Your code here
        for sentence in corpus:
            list_unigrams = get_ngrams(sentence, 1)
            for unigram in list_unigrams:
                if unigram not in self.unigramcounts:
                    self.unigramcounts[unigram] = 1
                else:
                    self.unigramcounts[unigram] += 1

            list_bigrams = get_ngrams(sentence, 2)
            for bigram in list_bigrams:
                if bigram not in self.bigramcounts:
                    self.bigramcounts[bigram] = 1
                else:
                    self.bigramcounts[bigram] += 1

            list_trigrams = get_ngrams(sentence, 3)
            for trigram in list_trigrams:
                if trigram not in self.trigramcounts:
                    self.trigramcounts[trigram] = 1
                else:
                    self.trigramcounts[trigram] += 1

            self.wordcount += len(sentence)
            self.sentencecount += 1


    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        trigram_count = self.trigramcounts[trigram]
        bigram = (trigram[0], trigram[1])
        if bigram == ('START', 'START'):
            bigram_count = self.sentencecount
        else:
            bigram_count = self.bigramcounts[(trigram[0], trigram[1])]
        return trigram_count / bigram_count


    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        bigram_count = self.bigramcounts[bigram]
        unigram_count = self.unigramcounts[(bigram[0])]
        return bigram_count / unigram_count


    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

        unigram_count = self.unigramcounts(unigram)
        return unigram_count/self.wordcounts


    def generate_sentence(self,t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        result = ["START", "START"]
        for i in range(0, t - 1):
            bigram = (result[-2], result[-1])
            dict_possible = {}
            for trigram in self.trigramcounts:
                if (trigram[0], trigram[1]) == bigram:
                    dict_possible[trigram[2]] = self.raw_trigram_probability(trigram)

            keys = [k for k in dict_possible.keys()]
            values = [int(v * 1000000) for v in dict_possible.values()]
            chosen = random.choices(keys, weights=values, k=1)[0]
            result.append(chosen)

            if chosen == "STOP":
                break

        # processing START and STOP as required
        if result[-1] != "STOP":
            result.append("STOP")
        result.pop(0)
        result.pop(0)

        return result


    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        # for a trigram (u, v, w)
        # q(w|u, v) = λ1 × qML(w|u, v) + λ2 × qML(w|v) + λ3 × qML(w)
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        p = lambda1 * self.raw_trigram_probability(trigram) + \
            lambda2 * self.raw_bigram_probability(trigram[1:]) + \
            lambda3 * self.raw_unigram_probability(trigram[2:])
        return p
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        return float("-inf")

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        return float("inf") 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
        
        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])
    print(model.generate_sentence())

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

