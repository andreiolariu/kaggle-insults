from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation
import numpy as np
                
class EnsembleGBC(object):
    '''
        neural network using several clasifiers as input
    '''
    
    def __init__(self, texts, classes, params, models):
        self.models = models
        self.params = params
            
        self.texts = texts
        self.classes = classes
        
        classes = np.array(classes)
        texts = np.array(texts)
        
        net_inputs = []
        expected_outputs = np.array([])
        
        print 'get training data'
        # get some training data
        for train, test in cross_validation.StratifiedKFold(classes, 5):
            texts_train = texts[train]
            classes_train = classes[train]
            # classes_train = list(classes_train)
            texts_test = texts[test]
            classes_test = classes[test]
            
            net_inputs_batch = []
            for model, params in self.models:
                m = model(texts_train, classes_train, *params)
                p = np.array(m.classify(texts_test))[np.newaxis]
                if len(net_inputs_batch):
                    net_inputs_batch = np.vstack([net_inputs_batch, p])
                else:
                    net_inputs_batch = p
            net_inputs_batch = net_inputs_batch.T
            
            expected_outputs = \
                    np.concatenate((expected_outputs, classes_test), axis=0)
            if len(net_inputs):
                net_inputs = \
                    np.vstack((net_inputs, net_inputs_batch))
            else:
                net_inputs = net_inputs_batch
    
        # init network
        self.gbc = GradientBoostingClassifier(max_depth=15, learn_rate=0.1)
        
        print 'train gbc'
        # train network
        #expected_outputs = expected_outputs[np.newaxis].T
        self.gbc.fit(net_inputs, expected_outputs)
    
    def classify(self, texts):
        print 'classify'
        net_inputs = []
        for model, params in self.models:
            m = model(self.texts, self.classes, *params)
            p = np.array(m.classify(texts))[np.newaxis]
            if len(net_inputs):
                net_inputs = np.vstack([net_inputs, p])
            else:
                net_inputs = p
        net_inputs = net_inputs.T
                
        pred_prob = self.gbc.predict_proba(net_inputs)
        predictions = []
        for pair in pred_prob:
            predictions.append(pair[1] - pair[0])
        predictions = np.array(predictions)
        predictions = (predictions + 1) / 2
        predictions[predictions > 1] = 1
        predictions[predictions < 0] = 0     
        return predictions
