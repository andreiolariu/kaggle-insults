import re
from math import sqrt

from scipy.sparse import csr_matrix
import numpy
import nltk

from static import *
NGRAM = 10 # Maximum ngram length
MINNGRAM = 4 # Minimum ngram length

class NLPDictCh(object):

    def __init__(self, texts):
        self.data = []
        for text in texts:
            self.data.append(self.tokenize(text))
        
        # Create dictionary
        freq = {}    
        for text in self.data:
            for i in xrange(len(text)):
                for j in xrange(i + MINNGRAM, i + NGRAM):
                    if j == len(text):
                        break
                    entry = text[i:j]
                    if entry not in freq:
                        freq[entry] = 1
                    else:
                        freq[entry] += 1
                    
        # Remove rare words
        min_freq = 1
        dictionary = set([w for w, f in freq.iteritems() if f >= min_freq])
        dictionary = list(dictionary)  
        # This seems to be the fastest way of creating and storing a dictionary
        self.dictionary = dict((dictionary[i], i) for i in xrange(len(dictionary)))
    
    def get_dictionary(self):
        return self.dictionary
    
    def tokenize(self, original_text):
        caps = sum([1 for ch in original_text if 'A' <= ch <= 'Z'])
        if caps:
            total = caps + sum([1 for ch in original_text if 'a' <= ch <= 'z'])
            ratio = float(caps) / total
        else:
            ratio = 0
        
        text = original_text.lower()
        text = text[1:-1]
        
        # Encodings....
        text = re.sub(r'\\\\', r'\\', text)
        text = re.sub(r'\\\\', r'\\', text)
        text = re.sub(r'\\x\w{2,2}', ' ', text)
        text = re.sub(r'\\u\w{4,4}', ' ', text)
        text = re.sub(r'\\n', ' . ', text)
        
        # Remove email adresses
        text = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', '_EMAIL', text)
        # Remove twitter user names
        text = re.sub(r'(\A|\s)@(\w+)', r'\1_TWUSER', text)
        # Remove urls
        text = re.sub(r'\w+:\/\/\S+', r'_URL', text)
        
        # Format whitespaces
        text = text.replace('"', ' ')
        text = text.replace('\'', ' ')
        text = text.replace('_', ' ')
        text = text.replace('-', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('\\n', ' ')
        text = re.sub(' +',' ', text)
        
        return text
        
    def feature_vectors(self, texts):
        '''
            Construct the feature vectors, given a list of texts
            Builds a sparse matrix
        '''
        # There 2 vectors will remember the where to add the 1s in the matrix
        row = []
        col = []
        for k in xrange(len(texts)):
            # Tokenize
            text = texts[k]
            text = self.tokenize(text)
            # Generate ngrams
            entries = set([])
            for i in xrange(len(text)):
                for j in xrange(i + MINNGRAM, i + NGRAM):
                    if j == len(text):
                        break
                    entry = text[i:j]
                    entries.add(entry)
            # Add ngrams to the sparse matrix 
            for entry in entries:
                if entry in self.dictionary:
                    row.append(k)
                    col.append(self.dictionary[entry])
        data = [1] * len(row)
        # Build the sparse matrix
        matrix = csr_matrix(
                (data, (row, col)), 
                shape=(len(texts), len(self.dictionary)),
                dtype=numpy.int8
        )
        return matrix
        
