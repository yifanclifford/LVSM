import numpy as np


def metric_dcg(pred, test, cut):
    r = np.array([x in test for x in pred])
    r[1:] = r[1:] / np.log2(np.arange(2, r.size + 1))
    res = [(r[0] + np.sum(r[1:c])) / c for c in cut]
    return res


def metric_recall(pred, test, cut):
    r = np.array([x in test for x in pred])
    res = [np.sum(r[:c] / len(test)) for c in cut]
    return res
