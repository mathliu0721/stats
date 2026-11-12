#判断模拟结果的精确度 用到伯努利分布 样本均值 中心极限定理

import numpy as np
from scipy import stats

rng = np.random.default_rng(42)  # 生成随机数据

n_runs = 1000

rejections = [] #空列表 用来记录每次模拟的拒绝原假设的结果

rank_sum_rejections = [] #空列表 用来记录每次模拟的拒绝原假设的结果

for r in range(n_runs):
    x = rng.normal(loc = 0,scale = 1,size = 20)
    y = rng.normal(loc = 0.5,scale = 1,size = 20) # 

    result = stats.ttest_ind(x,y,equal_var = False) # Welch t检验
    p_value = result.pvalue

    rejected = p_value < 0.05 # 判断是否拒绝原假设
    rejections.append(rejected) # 将结果添加到列表中

    rank_sum_result = stats.ranksums(x,y) # Wilcoxon符号秩检验
    rank_sum_p_value = rank_sum_result.pvalue

    rank_sum_rejected = rank_sum_p_value < 0.05 # 判断是否拒绝原假设
    rank_sum_rejections.append(rank_sum_rejected) # 将结果添加到列表中

power = np.mean(rejections) # 计算拒绝原假设的比例，即功效
rank_sum_power = np.mean(rank_sum_rejections) # 计算拒绝原假设的比例，即功效

mc_se = np.sqrt(power*(1-power)/n_runs) # 计算蒙特卡洛标准误差

lower_w = power - 1.96*mc_se # 计算置信区间下限
upper_w = power + 1.96*mc_se # 计算置信区间上限

print("试验次数",len(rejections))
print("前五轮是否拒绝",rejections[:5])
print("拒绝次数",sum(rejections))
print("估计功效",power)

print("Wilcoxon符号秩检验功效",rank_sum_power)

print("蒙特卡洛标准误差",mc_se)
print("95%置信区间",lower_w,upper_w)