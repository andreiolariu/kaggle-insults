import scipy.integrate
import numpy as np

def CalculateRoc(target_labels, predicted_labels):
    """Calculate the points of the ROC curve from a set of labels and evaluations
    of a classifier.
    
    Uses the single-pass efficient algorithm of Fawcett (2006). This assumes a
    binary classification task.
    
    :param target_labels: Ground-truth label for each instance.
    :type target_labels: 1D ndarray of float
    :param predicted_labels: Predicted label for each instance.
    :type predicted_labels: 1D ndarray of float
    :returns: Points on the ROC curve
    :rtype: 1D ndarray of float
    
    .. seealso::
    :func:`sklearn.metrics.roc_curve`
    
    """
    def iterator():
        num_pos = len(target_labels[ target_labels == 1 ])
        num_neg = len(target_labels) - num_pos
        i = predicted_labels.argsort()[::-1]
        fp = tp = 0
        last_e = -np.inf
        for l, e in zip(target_labels[i], predicted_labels[i]):
            if e != last_e:
                yield (fp / float(num_neg), tp / float(num_pos))
                last_e = e
            if l == 1:
                tp += 1
            else:
                fp += 1
        yield (fp / float(num_neg), tp / float(num_pos))
  
    return np.array(list(iterator()))

def calculate_auc(targets, predicted):
    targets = np.array(targets)
    predicted = np.array(predicted)
    points = CalculateRoc(targets, predicted)
    p = points.T
    return scipy.integrate.trapz(p[1], p[0])
