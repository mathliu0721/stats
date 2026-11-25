"""
第十个实验会检查：把正态数据换成右偏数据后，同样的 95% t 置信区间还能覆盖真实均值多少次
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

true_mean = 0
sample_sizes = [20, 80]
n_runs = 1000
confidence = 0.95

for sample_size in sample_sizes:
    critical_value = stats.t.ppf(
        (1 + confidence)/2,
        df = sample_size - 1
    )

    covered_results = []
    interval_widths = []

    for run in range(n_runs):
        x = rng.exponential(scale = 1, size = sample_size) - 1 + true_mean

        sample_mean = np.mean(x)
        sample_std = np.std(x, ddof = 1)
        standard_error = sample_std / np.sqrt(sample_size)

        lower = sample_mean - critical_value * standard_error
        upper = sample_mean + critical_value * standard_error

        covered = lower <= true_mean <= upper
        covered_results.append(covered)
        interval_widths.append(upper - lower)

        coverage = np.mean(covered_results)
        average_width = np.mean(interval_widths)

    print("覆盖次数:",sum(covered_results))
    print("估计覆盖率:",coverage)
    print("平均区间宽度:",average_width)