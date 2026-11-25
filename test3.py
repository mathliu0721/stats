#练习把模拟实验封装成函数。
#这样改变均值差时，调用一次函数就能得到一档功效
#也能解决之前遇到的“生成了新样本，检验却用了旧变量”的问题。

#先写Welch检验的函数
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def estimate_welch_power(rng, delta, n_runs):
    rejections = []

    for run in range(n_runs):
        x = rng.normal(loc = 0, scale = 1, size = 20)
        y = rng.normal(loc = delta, scale = 1, size = 20)

        result = stats.ttest_ind(x, y, equal_var = False)
        rejected = result.pvalue < 0.05
        rejections.append(rejected)

    power = np.mean(rejections)
    return power

rng = np.random.default_rng(42)

rate_0 = estimate_welch_power(rng, delta = 0, n_runs = 1000)
rate_05 = estimate_welch_power(rng, delta = 0.5, n_runs = 1000)

print("均值差为0时估计误报率:",rate_0)
print("均值差为0.5时估计功效:",rate_05)

def estimate_power(rng, delta, n_runs):
    welch_rejections = []
    rank_sum_rejections = []

    for run in range(n_runs):
        x = rng.normal(loc = 0, scale = 1, size = 20)
        y = rng.normal(loc = delta, scale = 1,size = 20)

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

deltas = [0, 0.25, 0.5, 0.75, 1.0]

welch_rates = []
rank_sum_rates = []

for delta in deltas:
    welch_rate,rank_sum_rate = estimate_power(rng, delta, n_runs = 1000)
    welch_rates.append(welch_rate)
    rank_sum_rates.append(rank_sum_rate)

    print("均值差为", delta, "时Welch检验拒绝比例:", welch_rate)
    print("均值差为", delta, "时秩和检验拒绝比例:", rank_sum_rate)

welch_power, rank_sum_power = estimate_power(rng, delta = 0.5, n_runs = 1000)

print("Welch检验功效:", welch_power)
print("秩和检验功效:", rank_sum_power)

plt.plot(deltas, welch_rates, label = "Welch t-test")
plt.plot(deltas, rank_sum_rates, label = "Wilcoxon rank_sum test")
plt.xlabel("Population mean difference")
plt.ylabel("Rejection rate")
plt.legend()
plt.show()

