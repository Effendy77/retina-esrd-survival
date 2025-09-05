<<<<<<< HEAD
# metrics helpers
=======

import numpy as np
from sklearn.metrics import roc_auc_score
def concordance_index(times, risks, events):
    n = len(times); num = 0.0; den = 0.0
    for i in range(n):
        for j in range(n):
            if times[i] < times[j] and events[i] == 1:
                den += 1
                if risks[i] < risks[j]:
                    num += 1
                elif risks[i] == risks[j]:
                    num += 0.5
    return num / den if den > 0 else 0.0
def time_dependent_auc(times, events, scores, t_cut):
    y = ((events == 1) & (times <= t_cut)).astype(int)
    if len(np.unique(y)) < 2: return float("nan")
    return roc_auc_score(y, scores)
>>>>>>> 2fc3584161108c072c8f40e607489f139530620a
