"""
第八个实验研究右偏数据下，两种检验的误报率和功效，写在 test8.py。这次继续使用第七题的功效差和模拟标准误

我们用指数分布构造右偏数据。它的数值集中在较小的一端,右侧拖着长尾。NumPy 用 rng.exponential(scale=1, size=...) 抽样，其中 scale 是尺度参数

当 \(E\sim\mathrm{Exp}(1)\) 时，
\[\mathbb E[E]=1, \qquad \operatorname{Var}(E)=1.\]
因此令 \(X=E-1\)，就得到均值为 0、方差为 1 的右偏分布。减去 1 只会平移分布，右偏的形状仍然保留
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def estimate_power(rng, delta, n_runs, sample_size, alpha, distribution):
    welch_rejections = []
    rank_sum_rejections = []

    for run in range(n_runs):
        if distribution == "normal":
            x = rng.normal(loc=0, scale=1, size=sample_size)
            y = rng.normal(loc=delta, scale=1, size=sample_size)
        
        elif distribution == "t3":
            x = rng.standard_t(df=3, size=sample_size) / np.sqrt(3)
            y = rng.standard_t(df=3, size=sample_size) / np.sqrt(3) + delta

        elif distribution == "exponential":
            x = rng.exponential(scale = 1, size = sample_size) - 1
            y = rng.exponential(scale = 1, size = sample_size) - 1 + delta
        
        else:
            raise ValueError("distribution 必须是 normal t3 or exponential")
            

        result = stats.ttest_ind(x, y, equal_var = False)
        welch_rejected = result.pvalue < alpha
        welch_rejections.append(welch_rejected)

        rank_sum_result = stats.ranksums(x, y)
        rank_sum_rejected = rank_sum_result.pvalue < alpha
        rank_sum_rejections.append(rank_sum_rejected)

    welch_results = np.array(welch_rejections, dtype = int)
    rank_sum_result = np.array(rank_sum_rejections, dtype = int)

    difference = rank_sum_result - welch_results

    gap = np.mean(difference)
    gap_se = np.std(difference,ddof = 1)/np.sqrt(n_runs)

    welch_power = np.mean(welch_rejections)
    rank_sum_power = np.mean(rank_sum_rejections)

    return welch_power, rank_sum_power, gap, gap_se

rng = np.random.default_rng(42)
deltas = [0, 0.5]

for delta in deltas:
    welch_rate, rank_sum_rate, gap, gap_se = estimate_power(
        rng,
        delta = delta,
        n_runs = 1000,
        sample_size = 20,
        alpha = 0.05,
        distribution = "exponential"
    )
    lower = gap - 1.96 * gap_se
    upper = gap + 1.96 * gap_se

    print("总体均值差", delta)
    print("Welch 拒绝比例", welch_rate)
    print("秩和检验拒绝比例", rank_sum_rate)
    print("拒绝比例之差（秩和减 Welch）", gap)
    print("差值的近似 95% 区间", lower, upper)
    print()