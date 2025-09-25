import numpy as np
# Note: do not import additional libraries to implement these functions


def compute_confusion_matrix(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the confusion matrix. The confusion
    matrix for a binary classifier would be a 2x2 matrix as follows:

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]

    You do not need to implement confusion matrices for labels with more
    classes. You can assume this will always be a 2x2 matrix.

    Compute and return the confusion matrix.

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels

    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    n_trueneg = 0
    n_falsepos = 0
    n_falseneg = 0
    n_truepos = 0
    
    for i in range(predictions.shape[0]):
        actual_val = actual[i]
        predicted_val = predictions[i]
        if actual_val==0 and predicted_val==0:
            n_trueneg+=1
        elif actual_val==0 and predicted_val==1:
            n_falsepos+=1
        elif actual_val==1 and predicted_val==0:
            n_falseneg+=1
        elif actual_val==1 and predicted_val==1:
            n_truepos+=1
            
    return np.array([[n_trueneg, n_falsepos],
                     [n_falseneg, n_truepos]])


def compute_accuracy(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the accuracy:

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        accuracy (float): accuracy
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    confusion_mat = compute_confusion_matrix(actual, predictions)
    total = np.sum(confusion_mat)
    total_right = np.trace(confusion_mat)
    if total == 0:
        return np.nan
    else:
        return total_right / total


def compute_precision_and_recall(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the precision and recall:

    https://en.wikipedia.org/wiki/Precision_and_recall

    You MUST account for edge cases in which precision or recall are undefined
    by returning np.nan in place of the corresponding value.

    Hint: implement and use the compute_confusion_matrix function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output a tuple containing:
        precision (float): precision
        recall (float): recall
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    conf_mat = compute_confusion_matrix(actual, predictions)
    if (conf_mat[1,1] + conf_mat[0,1]) == 0:
        precision = np.nan
    else:
        precision = conf_mat[1,1] / (conf_mat[1,1] + conf_mat[0,1])
        
    if (conf_mat[1,1] + conf_mat[1,0]) == 0:
        recall = np.nan
    else:
        recall = conf_mat[1,1] / (conf_mat[1,1] + conf_mat[1,0])
    return (precision, recall)

def compute_f1_measure(actual, predictions):
    """
    Given predictions (an N-length numpy vector) and actual labels (an N-length
    numpy vector), compute the F1-measure:

    https://en.wikipedia.org/wiki/Precision_and_recall#F-measure

    Because the F1-measure is computed from the precision and recall scores, you
    MUST handle undefined (NaN) precision or recall by returning np.nan. You
    should also consider the case in which precision and recall are both zero.

    Hint: implement and use the compute_precision_and_recall function!

    Args:
        actual (np.array): predicted labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        f1_measure (float): F1 measure of dataset (harmonic mean of precision and
        recall)
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = compute_precision_and_recall(actual, predictions)
    if ((precision == np.nan or recall == np.nan) or (precision == 0 and recall == 0)):
        return np.nan
    return 2 * (precision * recall) / (precision + recall)
