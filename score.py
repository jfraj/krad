"""
Routines to evaluate the score of a given prediction, following the
Kaggle defined metric: http://www.kaggle.com/c/how-much-did-it-rain/details/evaluation.
"""
import numpy as N
from scipy.stats import poisson
import matplotlib.pyplot as plt

def heaviside(rain_value):
    """
    Heaviside function, turning a single value into a step function
    input:
        - rain_value: single float representing the amount of rain in mm.
    output:
        values of the step function on the domain 0 <= x <= 69 mm.
    """
    x = N.arange(0,70)
    return N.where(N.round(rain_value) <= x, 1., 0)

def poisson_cumul(rain_value):
    """
    Poisson cumulative function as a smoother heaviside
    """
    x = N.arange(0,70)
    if rain_value < 1:
        return N.where(N.round(rain_value) <= x, 1., 0)
    return poisson.pmf(x, N.round(rain_value)).cumsum()

func_dic = {'heaviside': heaviside, 'poisson': poisson_cumul}


def kaggle_metric(predictions, exact_values, test_function='heaviside'):
    """Evaluate the score using the kaggle formula.

    input:
        - predictions : array of floats, dimension [ N,],
          with N the number of distinct items
        - exact_values: array of floats, dimension [ N,],
          number of data
    ouptut:
        - score: normalized score, following the Kaggle metric.
    """

    # Setting the test function
    test_function = func_dic[test_function]
    norm = 70.*len(predictions)

    score = 0.
    for p, e in zip(predictions, exact_values):
        score += N.sum((test_function(p)-heaviside(e))**2)

    return score/norm

def test_submission(predictions, exact_values, pred_id, exact_id):
    """Plots comparison of predictions with some values (shown as heavyside).

    input:
        - predictions : array of floats, dimension [ N, 70],
          with N the number of distinct items
        - exact_values: array of floats, dimension [ N,],
          number of data
    ouptut:
        - score: normalized score, following the Kaggle metric.
    """

    # Setting the test function
    norm = 70.*len(predictions)

    fig = plt.figure()
    score = 0.
    showcdf = True
    for p, e, pid, eid in zip(predictions, exact_values, pred_id, exact_id):
        iscore = N.sum((p-heaviside(e))**2)
        score += iscore
        if showcdf:
            print(p)
            print(e)
            print('Score increment={}'.format(iscore))
            fig.clear()
            plt.plot(p, c='b', linewidth=3, label='pred id={}'.format(pid))
            plt.plot(heaviside(e), c='r', linestyle='--', linewidth=3, label='true id={}'.format(eid))
            plt.legend(loc='best')
            plt.ylim([0,1.05])
            fig.canvas.draw()
            fig.show()
            if raw_input('Press enter to show next...') != '':
                showcdf = False

    return score/norm


def kaggle_score(clf, X_test, Y_true, test_function='heaviside'):
    """Scoring function to be used by sklearn evaluator.

    See this link for definition
    http://scikit-learn.org/dev/modules/model_evaluation.html#scoring-objects-defining-your-scoring-rules
    """
    return -kaggle_metric(clf.predict(X_test), Y_true, test_function)
