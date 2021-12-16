
from keras.backend import sum as kSum
from keras.backend import clip as kClip
from keras.backend import round as kRound
from keras.backend import mean as kMean
from keras.backend import epsilon as kEpsilon


def fbeta(y_true, y_pred, beta=2):
    # clip predictions
    y_pred = kClip(y_pred, 0, 1)
    # calculate elements
    tp = kSum(kRound(kClip(y_true * y_pred, 0, 1)), axis=1)
    fp = kSum(kRound(kClip(y_pred - y_true, 0, 1)), axis=1)
    fn = kSum(kRound(kClip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + kEpsilon())
    # calculate recall
    r = tp / (tp + fn + kEpsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = kMean((1 + bb) * (p * r) / (bb * p + r + kEpsilon()))
    return fbeta_score


def alpha(y_true, y_pred):
    y_pred = kClip(y_pred, 0, 1)
    tp = kSum(kRound(kClip(y_true * y_pred, 0, 1)), axis=1)
    # fp = kSum(kRound(kClip(y_pred - y_true, 0, 1)), axis=1)
    fn = kSum(kRound(kClip(y_true - y_pred, 0, 1)), axis=1)
    return tp / (tp + fn)
