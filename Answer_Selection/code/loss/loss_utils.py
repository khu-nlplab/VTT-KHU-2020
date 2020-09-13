import numpy as np
from itertools import combinations

def get_combinations(n, k):
    model_idx = set(np.arange(n))
    assigns = list(combinations(model_idx, k))
    not_assigns = []
    for cbn in assigns:
        not_assigns.append(model_idx - set(cbn))
        assert model_idx == (set.union(set(cbn), not_assigns[-1])), \
            "Union of assign and not-assign should be same with model_idx"

    return assigns, not_assigns