import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            # logL = self.score(self.X,self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
                # print("model created for {} with {} states and logL = {}".format(self.this_word, num_states, logL))                
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        word_sequences = self.sequences
        n=self.min_n_components
        BIC = {}        
        while (n <= self.max_n_components):
            # Use try/except construct to catch ValueErrors
            try:          
                # train a model based on current number of components
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                # calculate log loss for the model
                logL = model.score(self.X, self.lengths)
                # Calculate number of free parameters and log of the number of examples
                p = n*(n-1)
                logN = np.log(len(self.X))
                # Calculate BIC score using provided calculation and above model parameters
                BIC[n] = -2 * logL + p*logN
            except ValueError:
                pass
            n+=1

        # determine the number of components with the minimum BIC score
        best_num_components = min(BIC,key=BIC.get)
        # return best model
        return self.base_model(best_num_components)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        word_sequences = self.sequences
        n=self.min_n_components
        logProbX = {}
        DIC = {}        
        while (n <= self.max_n_components):
            # Use try/except construct to catch ValueErrors
            try:          
                # train a model based on current number of components
                model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                # calculate log probability for the model
                logProbX[n] = model.score(self.X, self.lengths)
            except ValueError:
                pass
            n+=1

        # Create a list of each of the choices (n_choices) and how many their are (M)        
        n_choices = list(logProbX.keys())
        M = len(n_choices)
        # Calculate DIC score using provided calculation and above model parameters        
        for i in n_choices:
            DIC[i] = logProbX[i] - (1/(M-1))*sum([logProbX[j] for j in n_choices if i!=j])

        # determine the number of components with the minimum DIC score
        best_num_components = min(DIC,key=DIC.get)
        # return best model
        return self.base_model(best_num_components)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)        

        # TODO implement model selection using CV
        word_sequences = self.sequences
        n=self.min_n_components        
        split_method = KFold(min(3,len(word_sequences))) 
        avgLogL={}
        while (n <= self.max_n_components):
            # Use try/except construct to catch ValueErrors
            try:
                sum_logL=0
                numCrossVals=0                
                # split word_sequences into training and testing sets for k folds
                for cv_train_idx, cv_test_idx in split_method.split(word_sequences):
                    X_train, lengths_train = combine_sequences(cv_train_idx, word_sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, word_sequences)
                    # train a submodel based on current train/test split
                    submodel = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    # calculate log loss for current submodel
                    logL = submodel.score(X_test,lengths_test)
                    # add log loss to cumulative log loss for this number of components and increment counter
                    sum_logL += logL
                    numCrossVals += 1
                # Calculate average log loss for the k folds                
                avgLogL[n] = (sum_logL / numCrossVals)
                # print("avgLogLoss for %d components is %d" % (n,avgLogL[n]))
            except ValueError:
                pass
            n+=1

        # determine the number of components with the minimum log loss and return it
        best_num_components = min(avgLogL,key=avgLogL.get)
        # return model with optimal number of components
        return self.base_model(best_num_components)        

# For testing purposes
if __name__ == "__main__":
    from  asl_test_model_selectors import TestSelectors
    test_model = TestSelectors()
    test_model.setUp()
    test_model.test_select_constant_interface()
    test_model.test_select_cv_interface()
    test_model.test_select_bic_interface()
    test_model.test_select_dic_interface()