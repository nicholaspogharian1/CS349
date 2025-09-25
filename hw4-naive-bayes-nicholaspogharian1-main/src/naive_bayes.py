import numpy as np
import warnings

from src.utils import softmax


class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when computing p(x|y).
                If the word "jackpot" appears `k` times across all documents with
                label y=1, we will instead record `k + self.smoothing`. Then
                `p("jackpot" | y=1) = (k + self.smoothing) / Z`, where Z is a
                normalization constant that accounts for adding smoothing to
                all words.
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Using self.alpha and self.beta, compute the probability p(y | X[i, :])
            for each row X[i, :] of X.  While you will have used log
            probabilities internally, the returned array should be
            probabilities, not log probabilities.

        See equation (9) in `naive_bayes.pdf` for a convenient way to compute
            this using your self.alpha and self.beta. However, note that
            (9) produces unnormalized log probabilities; you will need to use
            your src.utils.softmax function to transform those into probabilities
            that sum to 1 in each row.

        Args:
            X: a sparse matrix of shape `[n_documents, vocab_size]` on which to
               predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                np.sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"
        
        #sum should be Nx2
        if type(X) != np.ndarray:
            py0 = np.sum(np.nan_to_num(X.toarray()*self.beta[:,0]),axis=1)
            py1 = np.sum(np.nan_to_num(X.toarray()*self.beta[:,1]),axis=1)
        else:
            py0 = np.sum(np.nan_to_num(X*self.beta[:,0]),axis=1)
            py1 = np.sum(np.nan_to_num(X*self.beta[:,1]),axis=1)
        y0 = py0.reshape(py0.shape[0],1)
        y1 = py1.reshape(py1.shape[0],1)
        probs = self.alpha + np.concatenate((y0, y1), axis=1)
        return softmax(probs)

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        You should store log probabilities to avoid underflow.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        See equations (10) and (11) in `naive_bayes.pdf` for the math necessary
            to compute your alpha and beta.

        self.alpha should be set to contain the marginal probability of each class label.

        self.beta should be set to the conditional probability of each word
            given the class label: p(w_j | y_i). This should be an array of shape
            [n_vocab, n_labels].  Remember to use `self.smoothing` to smooth word counts!
            See __init__ for details. If we see M total words across all N documents with
            label y=1, have a vocabulary size of V words, and see the word "jackpot" `k`
            times, then: `p("jackpot" | y=1) = (k + self.smoothing) / (M + self.smoothing *
            V)` Note that `p("jackpot" | y=1) + p("jackpot" | y=0)` will not sum to 1;
            instead, `sum_j p(word_j | y=1)` will sum to 1.

        Hint: when self.smoothing = 0, some elements of your beta will be -inf.
            If `X_{i, j} = 0` and `\beta_{j, y_i} = -inf`, your code should
            compute `X_{i, j} \beta_{j, y_i} = 0` even though numpy will by
            default compute `0 * -inf` as `nan`.

            This behavior is important to pass both `test_smoothing` and
            `test_tiny_dataset_a` simultaneously.

            The easy way to do this is to leave `X` as a *sparse array*, which
            will solve the problem for you. You can also explicitly define the
            desired behavior, or use `np.nonzero(X)` to only consider nonzero
            elements of X.

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None; sets self.alpha and self.beta
        """
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size
        
        
        #calculate alpha
        p0 = np.log(1/n_docs * np.count_nonzero(np.equal(y, 0)))
        p1 = np.log(1/n_docs * np.count_nonzero(np.equal(y, 1)))
        self.alpha = np.array([p0,p1])
        
        #calculate beta
        numerator0 = np.sum(X[np.equal(y, 0)], axis=0) + self.smoothing
        numerator1 = np.sum(X[np.equal(y, 1)], axis=0) + self.smoothing
        denominator0 = np.sum(numerator0)
        denominator1 = np.sum(numerator1)
        beta0 = numerator0/denominator0
        beta1 = numerator1/denominator1
        
        self.beta = np.asarray(np.log(np.concatenate((beta0.reshape(vocab_size,1),beta1.reshape(vocab_size,1)), axis=1)))

    def likelihood(self, X, y):
        r"""
        Using fit self.alpha and self.beta, compute the log likelihood of the data.
            You should use logs to avoid underflow.
            This function should not use unlabeled data. Wherever y is NaN,
            that label and the corresponding row of X should be ignored.

        Equation (5) in `naive_bayes.pdf` contains the likelihood, which can be written:

            \sum_{i=1}^N \alpha_{y_i} + \sum_{i=1}^N \sum_{j=1}^V X_{i, j} \beta_{j, y_i}

            You can visualize this formula in http://latex2png.com

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: the log likelihood of the data
        """
        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2
        term1 = np.count_nonzero(np.equal(y, 0))*self.alpha[0] + np.count_nonzero(np.equal(y, 1))*self.alpha[1]
        #build term 2
        if type(X) != np.ndarray:
            Xmod0pre = (X[np.equal(y,0)].toarray() * self.beta[:,0])
            Xmod1pre = (X[np.equal(y,1)].toarray() * self.beta[:,1])
        else:
            Xmod0pre = (X[np.equal(y,0)] * self.beta[:,0])
            Xmod1pre = (X[np.equal(y,1)] * self.beta[:,1])
        Xmod0post = np.nan_to_num(Xmod0pre)
        Xmod1post = np.nan_to_num(Xmod1pre)
        Xmod0 = np.sum(Xmod0post)
        Xmod1 = np.sum(Xmod1post)
        term2 = Xmod0 + Xmod1
        return (term1+term2)

        
