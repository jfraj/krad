"""
Routines to evaluate the score of a given prediction, following the
Kaggle defined metric: http://www.kaggle.com/c/how-much-did-it-rain/details/evaluation.
"""
import numpy as N
from scipy.stats import poisson

def heaviside(rain_value):
    """
    Heaviside function, turning a single value into a step function
    input:
        - rain_value: single float representing the amount of rain in mm.
    output:
        values of the step function on the domain 0 <= x <= 69 mm.
    """
    x = N.arange(0,70)
    return N.where(rain_value <= x, 1., 0)

def poisson_cumul(rain_value):
    """
    Poisson cumulative function as a smoother heaviside
    """
    x = N.arange(0,70)
    if rain_value < 1:
        return N.where(rain_value <= x, 1., 0)
    return poisson.pmf(x, rain_value).cumsum()

func_dic = {'heaviside': heaviside, 'poisson': poisson_cumul}

def kaggle_metric(predictions, exact_values, test_function = 'heaviside'):
    """Evaluate the score using the kaggle formula.

    input:
        - predictions : array of floats, dimension [ N, 70], with N the number of distinct items
        - exact_values: array of floats, dimension [ N, 70], representing the exact value step functions.
    ouptut:
        - score: normalized score, following the Kaggle metric.
    """
    ## Setting the test function


    test_function = func_dic[test_function]
    norm = 70.*len(predictions)

    score = 0.
    for p,e in zip(predictions,exact_values):
        score += N.sum((test_function(p)-heaviside(e))**2)
        #score += N.sum((heaviside(p)-heaviside(e))**2)

    return score/norm

def kaggle_score(clf, X_test, Y_true, test_function = 'heaviside'):
    """Scoring function to be used by sklearn evaluator.

    See this link for definition
    http://scikit-learn.org/dev/modules/model_evaluation.html#scoring-objects-defining-your-scoring-rules
    """
    return kaggle_metric(clf.predict(X_test), Y_true, test_function)
