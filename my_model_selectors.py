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
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
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
                num_features = len(self.X[0])
                p = n*(n-1) + 2*num_features*n
                logN = np.log(len(self.X))
                # Calculate BIC score using provided calculation and above model parameters
                BIC[n] = -2 * logL + p*logN
                # print("BIC for %d components is %d" % (n,BIC[n]))                
            except ValueError:
                pass
            n+=1

        try:
            # determine the number of components with the minimum BIC score
            best_num_components = min(BIC,key=BIC.get)
        except ValueError:
            # if none of the components have passable num components, pass min components to base_model
            best_num_components = self.min_n_components
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
        thisword = self.this_word
        allwords = (self.words).keys()       
        n=self.min_n_components 
        DIC = {}        
        while (n <= self.max_n_components):
            logL = {}
            validwords = []
            # train and score a model for every word
            for i in allwords:                
                X_i, lengths_i = self.hwords[i]
                # Use try/except construct to catch ValueErrors
                try:          
                    # train a model based on current number of components
                    model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_i, lengths_i)
                    # calculate log probability for the model
                    logL[i] = model.score(X_i, lengths_i)       
                    validwords.append(i)
                except ValueError:
                    pass
            # Try calculating DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
            try:
                M=len(validwords)                
                DIC[n] = logL[thisword] - (1/(M-1))*sum([logL[j] for j in validwords if j!=thisword])
            # Catch errors if there is no valid model for thisword with n components
            except KeyError:
                DIC[n] = float("-inf")            
            n+=1
        try:        
            # determine the number of components with the maximum DIC score
            best_num_components = max(DIC,key=DIC.get)
        except ValueError:
            # if none of the components have passable num components, pass min components to base_model
            best_num_components = self.min_n_components            
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
        # If the word has less than 3 examples, skip CV and return base model with n==3
        if len(word_sequences) < 3:
            return self.base_model(3)
        split_method = KFold() 
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

        try:
            # determine the number of components with the max log loss and return it
            best_num_components = max(avgLogL,key=avgLogL.get)            
        except ValueError:
            # if none of the components have passable num components, pass min components to base_model
            best_num_components = self.min_n_components            
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