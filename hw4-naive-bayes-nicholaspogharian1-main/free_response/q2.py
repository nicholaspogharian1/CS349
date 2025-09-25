import numpy as np

from free_response.data import build_dataset
from src.naive_bayes_em import NaiveBayesEM


def q2a():
    """
    First, fit a NaiveBayesEM model to the dataset.  Then, using your fit
    model's `beta` parameters, define:
      `f(j) = p(w_j | y = 1, beta) - p(w_j | y = 0, beta)`

    as the difference in "party affiliation" of the `j`th word, which takes
    positive values for words that are more likely in speeches by Republican
    presidents and takes negative values for words more likely in speeches by
    Democratic presidents.

    Compute `f(j)` for each word `w_j` in the vocabulary. Please use
    probabilities, not log probabilities.

    Hint: `f(j)` should be between -1 and 1 for all `j` (and quite close to 0)

    You will use these `f(j)` values to answer FRQ 2a.
    """
    # Load the dataset; fit the NB+EM model
    data, labels, speeches, vocab = build_dataset(
        "data", num_docs=100, max_words=2000, vocab_size=1000)
    nbem = NaiveBayesEM(max_iter=10)
    nbem.fit(data, labels)

    #get to actual probs from log probs
    probs = np.exp(nbem.beta)
    diffs = probs[:,1]-probs[:,0]
    sort = np.argsort(diffs) #sorted indices
    print('5 highest words and their values')
    print(vocab[sort[-5:]])
    print(diffs[sort[-5:]])
    print('5 lowest words and their values')
    print(vocab[sort[0:5]])
    print(diffs[sort[0:5]])


def q2b():
    """
    Helper code for the Free Response Q2b
    You shouldn't need to edit this function
    """
    # Load the dataset; fit the NB+EM model
    data, labels, speeches, vocab = build_dataset(
        "data", num_docs=100, max_words=2000, vocab_size=1000)
    isfinite = np.isfinite(labels)
    nbem = NaiveBayesEM(max_iter=10)
    nbem.fit(data, labels)

    # Use predict_proba to see output probabilities
    probs = nbem.predict_proba(data)[isfinite]
    preds = nbem.predict(data)
    correct = preds[isfinite] == labels[isfinite]

    # The model's "confidence" in its predicted output when right 
    right_label = labels[isfinite][correct].astype(int)
    prob_when_correct = probs[correct, right_label]

    # The model's "confidence" in its predicted output when wrong 
    incorrect = np.logical_not(correct)
    wrong_label = 1 - labels[isfinite][incorrect].astype(int)
    prob_when_incorrect = probs[incorrect, wrong_label]

    # Use these number to answer FRQ 2b
    print("When NBEM is correct:")
    print(prob_when_correct.tolist())
    print("When NBEM is incorrect:")
    print(prob_when_incorrect.tolist())


if __name__ == "__main__":
    q2a()
    q2b()
