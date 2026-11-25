import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def estimate_power(rng, delta, n_runs, sample_size):
    welch_rejections = []
    rank_sum_rejections = []

    for run in range(n_runs):
        x = rng.normal(loc = 0, scale = 1, size = sample_size)
        y = rng.normal(loc = delta, scale = 1, size = sample_size)

        result = stats.ttest_ind(x, y, equal_var = False)
        welch_rejected = result.pvalue < 0.05
        welch_rejections.append(welch_rejected)

        rank_sum_result = stats.ranksums(x, y)
        rank_sum_rejected = rank_sum_result.pvalue < 0.05
        rank_sum_rejections.append(rank_sum_rejected)

    welch_power = np.mean(welch_rejections)
    rank_sum_power = np.mean(rank_sum_rejections)

    return welch_power, rank_sum_power

rng = np.random.default_rng(42)

sample_sizes = [10, 20, 40, 80]
welch_rates = []
rank_sum_rates = []

for sample_size in sample_sizes:
    welch_rate, rank_sum_rate = estimate_power(rng, delta = 0.5, n_runs = 1000, sample_size = sample_size)

    welch_rates.append(welch_rate)
    rank_sum_rates.append(rank_sum_rate)

    print("每组样本量:",sample_size)
    print("Welch功效:",welch_rate)
    print("秩和检验功效:",rank_sum_rate)

plt.plot(sample_sizes, welch_rates, "o-", label="Welch t-test")
plt.plot(sample_sizes, rank_sum_rates, "s-", label="Wilcoxon rank-sum")

plt.xlabel("Sample size per group")
plt.ylabel("Estimated power")
plt.legend()
plt.show()

