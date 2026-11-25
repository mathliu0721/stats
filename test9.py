r"""
第九个实验会从假设检验转到置信区间,检查“95% 置信区间”在重复抽样中有多少次覆盖真实均值
在独立正态样本的设定下，总体均值的 \(t\) 置信区间为
\[\left[\bar X-t_{0.975,n-1}\frac{S}{\sqrt n},
\quad\bar X+t_{0.975,n-1}\frac{S}{\sqrt n}\right].\]
这里 \(\bar X\) 是样本均值，\(S\) 是样本标准差。95% 的含义是，按这种方法重复构造区间，长期约有 95% 的区间包含真实均值
"""

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

true_mean = 0
sample_size = 20
n_runs = 1000
confidences = [0.95, 0.99]

for confidence in confidences:
    critical_value = stats.t.ppf(
        (1 + confidence)/2,
        df = sample_size - 1
    )

    covered_results = []
    interval_widths = []

    for run in range(n_runs):
        x = rng.normal(loc = true_mean, scale = 1, size = sample_size)

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

