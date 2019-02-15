import numpy as np
from sklearn.metrics import make_scorer

def find_threshold_for_efficiency(a, e, w):
    if e < 0 or e > 1:
        raise ValueError("Efficiency e must be in [0, 1]")
    # Decreasing order
    idx = np.argsort(a)[::-1]
    a_sort = a[idx]
    if w is None:
        w = np.ones(a.shape)
    w_sort = w[idx]
    ecdf = np.cumsum(w_sort)
    if (ecdf[-1]) <= 0:
        raise ValueError("Total weight is < 0")

    target_weight_above_threshold = e * ecdf[-1]
    enough_passing = ecdf >= target_weight_above_threshold
    first_suitable = np.argmax(enough_passing)
    last_unsuitable_inv = np.argmin(enough_passing[::-1])
    if last_unsuitable_inv == 0:
        raise ValueError("Bug in code")
    last_unsuitable_plus = len(a) - last_unsuitable_inv
    return 0.5*(a_sort[first_suitable] + a_sort[last_unsuitable_plus])


def get_rejection_at_efficiency_raw(
        labels, predictions, weights, quantile):
    signal_mask = (labels >= 1)
    background_mask = ~signal_mask
    if weights is None:
        signal_weights = None
    else:
        signal_weights = weights[signal_mask]
    threshold = find_threshold_for_efficiency(predictions[signal_mask], 
                                              quantile, signal_weights)
    rejected_indices = (predictions[background_mask] < threshold)
    if weights is not None:
        rejected_background = weights[background_mask][rejected_indices].sum()
        weights_sum = np.sum(weights[background_mask])
    else:
        rejected_background = rejected_indices.sum()
        weights_sum = np.sum(background_mask)
    return rejected_background, weights_sum         


def get_rejection_at_efficiency(labels, predictions, threshold, sample_weight=None):
    rejected_background, weights_sum = get_rejection_at_efficiency_raw(
        labels, predictions, sample_weight, threshold)
    return rejected_background / weights_sum


def rejection90(labels, predictions, sample_weight=None):
    return get_rejection_at_efficiency(labels, predictions, 0.9, sample_weight=sample_weight)


rejection90_sklearn = make_scorer(
    get_rejection_at_efficiency, needs_threshold=True, threshold=0.9)


class Rej90Loss(object):
    def get_final_error(self, error, weight):
#         return error / (weight + 1e-38)
        return error

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        # approxes is list of indexed containers (containers with only __len__ and __getitem__ defined), one container
        # per approx dimension. Each container contains floats.
        # weight is one dimensional indexed container.
        # target is float.
        
        # weight parameter can be None.
        # Returns pair (error, weights sum)
#         approxes = [approxes[i] for i in range(len(approxes))]
#         print(len(approxes2),len(target),len(weight))
#         target = [target[i] for i in range(len(target))]
#         weight = [weight[i] for i in range(len(weight))]
        approx = approxes[0]
        approx = np.array([approx[i] for i in range(len(approx))])
        target = np.array([target[i] for i in range(len(target))])
        weight = np.array([weight[i] for i in range(len(weight))])
        score = get_rejection_at_efficiency(target,approx, 0.9, weight)
        return score, 1.

