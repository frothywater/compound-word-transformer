import numpy as np
from sklearn.model_selection import LeaveOneOut

from mgeval import core, utils

metrics_shape = {
    "total_used_pitch": (1,),
    "total_pitch_class_histogram": (12,),
    "pitch_class_transition_matrix": (12, 12),
    "pitch_range": (1,),
    "avg_pitch_shift": (1,),
    "total_used_note": (1,),
    "avg_IOI": (1,),
    "note_length_hist": (12,),
    "note_length_transition_matrix": (12, 12),
}
metrics = metrics_shape.keys()


def features(files, num_samples):
    result = {metric: np.zeros((num_samples,) + metrics_shape[metric]) for metric in metrics}
    indices = np.random.choice(len(files), num_samples, replace=False)
    for metric in metrics:
        count = 0
        for i in indices:
            feature = core.extract_feature(files[i])
            metric_result = getattr(core.metrics(), metric)(feature)
            result[metric][count] = metric_result
            count += 1
    return result


def cross_valid(set1, set2):
    loo = LeaveOneOut()
    num_samples = len(set1[list(metrics)[0]])
    loo.get_n_splits(np.arange(num_samples))
    result = np.zeros((num_samples, len(metrics), num_samples))
    for i, metric in enumerate(metrics):
        for _, test_index in loo.split(np.arange(num_samples)):
            result[test_index[0]][i] = utils.c_dist(set1[metric][test_index], set2[metric])
    return np.transpose(result, (1, 0, 2)).reshape(len(metrics), -1)


def mean_std(intra):
    mean = {metric: np.mean(intra[i]) for i, metric in enumerate(metrics)}
    std = {metric: np.std(intra[i]) for i, metric in enumerate(metrics)}
    return mean, std


def kl_divergence(intra, inter):
    return {metric: utils.kl_dist(intra[i], inter[i]) for i, metric in enumerate(metrics)}


def overlap_area(intra, inter):
    return {metric: utils.overlap_area(intra[i], inter[i]) for i, metric in enumerate(metrics)}

