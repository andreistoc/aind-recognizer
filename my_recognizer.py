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
    probabilities = []
    guesses = []

    # Guidance from mentor Chris:
    # Loop through all the words in test_set then score the word with every model and see which one gives the best score
    # Store all scores in a list and add that to the probabilities list
    # Also keep track of which one generates the highest score and look at what word that model represents
    # Add that word to the guesses list
    for X, lengths in test_set.get_all_Xlengths().values():
        best_score = float("-inf")
        best_word = None
        probability_dict = {}

        for word, model in models.items():
            try:
                score = model.score(X, lengths)
                probability_dict[word] = score
                if score > best_score:
                    best_score = score
                    best_word = word
            except:
                probability_dict[word] = float("-inf")

        probabilities.append(probability_dict)
        guesses.append(best_word)

    return probabilities, guesses
