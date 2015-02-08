"""
Routines to evaluate the score of a given prediction, following the
Kaggle defined metric: http://www.kaggle.com/c/how-much-did-it-rain/details/evaluation.
"""
import numpy as N


def heaviside(rain_value):
    """
    Heaviside function, turning a single value into a step function
    input:
        - rain_value: single float representing the amount of rain in mm.
    output:
        values of the step function on the domain 0 <= x <= 69 mm.
    """
    x = N.arange(0,70)
    P = N.where(rain_value <= x, 1., 0)
    return P

def kaggle_metric(predictions, exact_values):
    """
    Evaluate the score using the kaggle formula.
    input:
        - predictions : array of floats, dimension [ N, 70], with N the number of distinct items
        - exact_values: array of floats, dimension [ N, 70], representing the exact value step functions.
    ouptut:
        - score: normalized score, following the Kaggle metric.
    """
    norm = 70.*len(predictions)
                    
    score = 0.
    for p,e in zip(predictions,exact_values):
        score += N.sum((heaviside(p)-heaviside(e))**2)

    return score/norm
