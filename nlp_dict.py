import re
from math import sqrt

from scipy.sparse import csr_matrix
import numpy
import nltk

from static import *
NGRAM = 4 # Maximum ngram length

class NLPDict(object):

    def __init__(self, texts=[], dictionary=None):
        '''
            initializes the dictionary for creating feature vectors out of text
            if fresh init -> texts must be provided
            if initialized from previous data -> dictionary must be provided
        '''
        
        self.ngram = NGRAM
        self.stemmer = nltk.stem.PorterStemmer()
        
        if dictionary:
            self.dictionary = dictionary
            return
            
        #if len(texts) < 500:
        #    raise Exception('This Dictionary class must be initialized with a corpus of texts')
        
        self.data = []
        for text in texts:
            self.data.append(self.tokenize(text))
        
        # Create dictionary
        freq = {}    
        for item in self.data:
            for i in xrange(len(item['words'])):
                entry = ''
                for j in xrange(i, i + self.ngram):
                    if j == len(item['words']):
                        break
                    entry += item['words'][j]
                    if entry not in freq:
                        freq[entry] = 1
                    else:
                        freq[entry] += 1
                    entry += ' '
                    
        # TODO: remove stopwords?
        # My approach ignores frequencies - sligh improvement:
        # http://cs.northwestern.edu/~rms868/mlearn-395-22/sentiment-svm/results.html
        
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
        text = re.sub(r'[\w\-][\w\-\.]+@[\w\-][\w\-\.]+[a-zA-Z]{1,4}', '_EM', text)
        # Remove urls
        text = re.sub(r'\w+:\/\/\S+', r'_U', text)
        
        # Format whitespaces
        text = text.replace('"', ' ')
        text = text.replace('\'', ' ')
        text = text.replace('_', ' ')
        text = text.replace('-', ' ')
        text = text.replace('\n', ' ')
        text = text.replace('\\n', ' ')
        text = re.sub(' +',' ', text)
        
        clean_text = text
        
        text = text.replace('\'', ' ')
        text = re.sub(' +',' ', text)
        
        # Is somebody cursing?
        text = re.sub(r'([#%&\*\$]{2,})(\w*)', r'\1\2 _CR', text)
        # Remove repeated question marks
        text = re.sub(r'([^!\?])(\?{2,})(\Z|[^!\?])', r'\1 _BQ\n\3', text)
        # Remove repeated question marks
        text = re.sub(r'([^\.])(\.{2,})', r'\1 _SS\n', text)
        # Remove repeated exclamation (and also question) marks
        text = re.sub(r'([^!\?])(\?|!){2,}(\Z|[^!\?])', r'\1 _BX\n\3', text)
        # Remove single question marks
        text = re.sub(r'([^!\?])\?(\Z|[^!\?])', r'\1 _Q\n\2', text)
        # Remove single exclamation marks
        text = re.sub(r'([^!\?])!(\Z|[^!\?])', r'\1 _X\n\2', text)
        # Remove repeated (3+) letters: cooool --> cool, niiiiice --> niice 
        text = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL', text)
        # Do it again in case we have coooooooollllllll --> cooll
        text = re.sub(r'([a-zA-Z])\1\1+(\w*)', r'\1\1\2 _EL', text)
        # Remove smileys (big ones, small ones, happy or sad)
        text = re.sub(r' [8x;:=]-?(?:\)|\}|\]|>){2,}', r' _BS', text)
        text = re.sub(r' (?:[;:=]-?[\)\}\]d>])|(?:<3)', r' _S', text)
        text = re.sub(r' [x:=]-?(?:\(|\[|\||\\|/|\{|<){2,}', r' _BF', text)
        text = re.sub(r' [x:=]-?[\(\[\|\\/\{<]', r' _F', text)
        # Remove dots in words
        text = re.sub(r'(\w+)\.(\w+)', r'\1\2', text)
        
        # Split in phrases
        phrases = re.split(r'[;:\.()\n]', text)
        phrases = [re.findall(r'[\w%\*&#]+', ph) for ph in phrases]
        phrases = [ph for ph in phrases if ph]
        
        words = []
        for ph in phrases:
            words.extend(ph)
            
        # search for sequences of single letter words
        # like this ['f', 'u', 'c', 'k'] -> ['fuck']
        tmp = words
        words = []
        new_word = ''
        for word in tmp:
            if len(word) == 1:
                new_word = new_word + word
            else:
                if new_word:
                    words.append(new_word)
                    new_word = ''
                words.append(word)
        
        stemmed = [self.stemmer.stem(word) for word in words]
        
        # check if a curse was split into 2 words
        i = 0
        while i < len(words) - 1:
            if words[i] + words[i + 1] in stemmed_curse_words:
                words[i : i + 2] = [words[i] + words[i + 1]]
            else:
                i += 1
        
        return {'original': original_text,
            'words': words,
            'stemmed': stemmed,
            'ratio': ratio,
            'clean': clean_text}
        
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
            item = self.tokenize(text)
            # Generate ngrams
            entries = set([])
            for i in xrange(len(item['words'])):
                entry = ''
                for j in xrange(i, i + self.ngram):
                    if j == len(item['words']):
                        break
                    entry += item['words'][j]
                    entries.add(entry)
                    entry += ' '
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
        
