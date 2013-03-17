import time

import sklearn
from sklearn import cross_validation
import numpy as np

from svm import SVM
from naive import MNB
from ensemblegbc import EnsembleGBC
from ensemble import Ensemble
from auc import calculate_auc

class TestSVM(object):
    
    def __init__(self):
        pass
    
    @staticmethod
    def test(texts, classes, models, nn_params, folds=4):
        '''
            Check the performance on an SVM implementation,
            given a list of texts and their classes (negative/neutral/positive)
            Uses k-fold cross-validation (keeping in mind to divide the data
            appropriately, depending on the class)
        '''
        classes = np.array(classes)
        texts = np.array(texts)
        
        wrongs = []
        auc_sum = 0
        
        for train, test in cross_validation.StratifiedKFold(classes, folds):
            texts_train = texts[train]
            classes_train = classes[train]
            texts_test = texts[test]
            classes_test = classes[test]
            n = Ensemble(texts_train, classes_train, nn_params, models)
            predictions = n.classify(texts_test)
            predictions[predictions<0] = 0
            
            auc = calculate_auc(classes_test, predictions)
            print auc
            auc_sum += auc
            
            for i in range(len(texts_test)):
                if abs(classes_test[i] - predictions[i]) > 0.5:
                    wrongs.append((classes_test[i], predictions[i], texts_test[i]))
            
        '''
        import csv
        writer = open('wrongs.csv', 'w')
        for w in wrongs:
            writer.write('%s,%s,%s\n' % w)
        writer.close()
        '''
        
        return auc_sum / folds
    
    @staticmethod
    def test_model(texts, classes, model, folds=5):
        (model, params) = model
        classes = np.array(classes)
        texts = np.array(texts)
        auc_sum = 0
        
        for train, test in cross_validation.StratifiedKFold(classes, folds):
            texts_train = texts[train]
            classes_train = classes[train]
            texts_test = texts[test]
            classes_test = classes[test]
            s = model(texts_train, classes_train, *params)
            predictions = s.classify(texts_test)
            
            auc = calculate_auc(classes_test, predictions)
            auc_sum += auc
            
        return auc_sum / folds

