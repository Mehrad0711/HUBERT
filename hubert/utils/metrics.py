import numpy as np
from sklearn.metrics import confusion_matrix, r2_score
import sys

def class_acc(out, labels):
    outputs = np.argmax(out, axis=1)
    sys.stdout.flush()
    print('\n')
    print('confusion_matrix:\n', confusion_matrix(labels, outputs))
    print('\n')
    sys.stdout.flush()
    return np.sum(outputs == labels)

def reg_acc(out, labels):
    sys.stdout.flush()
    r2_value = r2_score(labels, out)
    print('\n')
    print('R2 score is:\n', r2_value)
    print('\n')
    sys.stdout.flush()
    return r2_value