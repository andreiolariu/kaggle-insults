import re

from static import *

from nlp_dict import NLPDict

class Dictionary(object):

    def __init__(self, texts, classes):
        pass
        
    def classify(self, texts):
        n = NLPDict(texts)
        results = []
        for item in n.data:
            text_words = set(item['stemmed'])
            if text_words.intersection(you_words) \
                    and text_words.intersection(stemmed_curse_words):
                rez = 0.5
                you_pos = [i for i, w in enumerate(item['stemmed']) \
                        if w in you_words]
                curse_pos = [i for i, w in enumerate(item['stemmed']) \
                        if w in stemmed_curse_words or \
                        re.sub(r'[^\w]', '', w) in stemmed_curse_words or \
                        sum([1 for f in freq_curse_words if f in w])]
                for p1 in you_pos:
                    for p2 in curse_pos:
                        rez += 0.13 / abs(p1 - p2)
                rez = min(rez, 1)
                results.append(rez)
            else:
                results.append(0)
        return results

