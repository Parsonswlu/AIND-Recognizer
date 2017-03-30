import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

    :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
    :param test_set: SinglesData object
    :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
    """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for this_word in test_set.get_all_sequences():
        X_test, lengths_test = test_set.get_item_Xlengths(this_word)
        logL = {}
        for this_key, this_model in models.items():
            try:
                logL[this_key] = this_model.score(X_test,lengths_test)
            except ValueError:
                logL[this_key] = float("-inf")
        probabilities.append(logL)
        guesses.append(max(logL,key=logL.get))
    return probabilities, guesses

# For testing purposes
if __name__ == "__main__":
    from  asl_test_recognizer import TestRecognize
    test_model = TestRecognize()
    test_model.setUp()
    test_model.test_recognize_probabilities_interface()
    test_model.test_recognize_guesses_interface()