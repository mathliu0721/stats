import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def estimate_power(rng, delta, n_runs, sample_size, alpha):
    welch_rejections = []
    rank_sum_rejections = []

    for run in range(n_runs):
        x = rng.normal(loc = 0, scale = 1, size = sample_size)
        y = rng.normal(loc = delta, scale = 1, size = sample_size)

        result = stats.ttest_ind(x, y, equal_var = False)
        welch_rejected = result.pvalue < alpha
        welch_rejections.append(welch_rejected)

        rank_sum_result = stats.ranksums(x, y)
        rank_sum_rejected = rank_sum_result.pvalue < alpha
        rank_sum_rejections.append(rank_sum_rejected)

    welch_power = np.mean(welch_rejections)
    rank_sum_power = np.mean(rank_sum_rejections)

    return welch_power, rank_sum_power

rng = np.random.default_rng(42)
alphas = [0.01, 0.05, 0.10]

for alpha in alphas:
    welch_false_alarm, rank_sum_false_alarm = estimate_power(
        rng,
        delta = 0,
        n_runs = 1000,
        sample_size = 20,
        alpha = alpha
    )

    welch_power, rank_sum_power = estimate_power(
        rng,
        delta = 0.5,
        n_runs = 1000,
        sample_size = 20,
        alpha = alpha
    )

    print("显著性水平:",alpha)
    print("Welch误报率:",welch_false_alarm)
    print("Welch功效:",welch_power)
    print("秩和检验误报率:",rank_sum_false_alarm)
    print("秩和检验功效:",rank_sum_power)
    print()


