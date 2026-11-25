"""
第六个实验研究数据分布变了，两种检验的表现会怎样变化
前面的数据都来自正态分布。这次加入重尾分布，它出现远离中心的极端数值的概率更大。我们用自由度为 3 的
t分布生成这类数据,NumPy 对应的函数是 rng.standard_t(df=3, size=...)
实验固定每组样本量为 20、显著性水平为 0.05,
每种情况模拟 1,000 轮。两种分布都检查 delta=0 时的误报率，以及 delta=0.5 时的功效。
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
        
        else:
            raise ValueError("distribution 必须是 normal 或 t3")
            

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
distributions = ["normal", "t3"]
deltas = [0, 0.5]

for distribution in distributions:
    for delta in deltas:
        welch_rate, rank_sum_rate = estimate_power(
             rng,
             delta = delta,
             n_runs = 1000,
             sample_size = 20,
             alpha = 0.05,
             distribution = distribution
        )
        
        print("分布:",distribution, "均值差:",delta)
        print("Welch拒绝比例:",welch_rate)
        print("秩和检验拒绝比例:",rank_sum_rate)
        print()