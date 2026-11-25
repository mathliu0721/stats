"""
第七个实验回答一个科研中常见的问题：两种方法的估计功效相差十几个百分点，这个差值的模拟误差有多大
在 test6.py 中已经保存了两种检验每轮的判断。这次沿用它们，在 test7.py 中计算每轮判断之差
固定使用标准化 \(t_3\) 数据、均值差 0.5、每组 20 个观测
把拒绝记为 1,不拒绝记为 0并约定
\[D_r=I_{\text{秩和},r}-I_{\text{Welch},r}.\]
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

    welch_results = np.array(welch_rejections, dtype = int)
    rank_sum_result = np.array(rank_sum_rejections, dtype = int)

    difference = rank_sum_result - welch_results

    gap = np.mean(difference)
    gap_se = np.std(difference,ddof = 1)/np.sqrt(n_runs)

    welch_power = np.mean(welch_rejections)
    rank_sum_power = np.mean(rank_sum_rejections)

    return welch_power, rank_sum_power, gap, gap_se

rng = np.random.default_rng(42)

welch_power, rank_sum_power, gap, gap_se = estimate_power(
    rng,
    delta = 0.5,
    n_runs = 1000,
    sample_size = 20,
    alpha = 0.05,
    distribution = "t3" 
    )

lower = gap - 1.96 * gap_se
upper = gap + 1.96 * gap_se

print("Welch 功效", welch_power)
print("秩和检验功效", rank_sum_power)
print("功效差（秩和减 Welch）", gap)
print("差值的蒙特卡洛标准误", gap_se)
print("差值的近似 95% 区间", lower, upper)


