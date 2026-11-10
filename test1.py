import numpy as np

rng = np.random.default_rng(42)  # 生成随机数据

x = rng.normal(loc = 0, scale = 1, size = 20) # 正态分布
y = rng.normal(loc = 0.5, scale = 1, size = 20)

print("x y的前五个观测",x[:5],"\n",y[:5])
print("x y的均值",x.mean(),"\n",y.mean()) # 求均值

def Welch():
    # Welch t检验
    from scipy import stats # scipy.stats模块中有很多统计学方法
    t_stats,p_value = stats.ttest_1samp(x,0)
    """
    先说明上步做的是单变量t检验，检验的原假设是x的均值为0，备择假设是x的均值不为0
    再说明一下stats.ttest_1samp()适用于单样本t检验，返回的t_stats是t统计量，p_value是对应的p值
    下面写两独立样本t检验，检验的原假设是x和y的均值相等，备择假设是x和y的均值不相等
    """
    t_stats,p_value = stats.ttest_ind(x,y,equal_var=False) # Welch t检验
    return t_stats, p_value

def Wilcoxon():
    # Wilcoxon符号秩检验 用独立样本秩和检验
    from scipy import stats
    t_stats,p_value = stats.ranksums(x,y) # 独立样本秩和检验
    return t_stats, p_value

if __name__ == "__main__":
    """
    下面利用最开始给出的x和y数据进行1000次Welch t检验和Wilcoxon符号秩检验，并输出结果
    """
    welch_t_stats = []
    welch_p_values = []
    wilcoxon_t_stats = []
    wilcoxon_p_values = []

    for i in range(1000):
        x = rng.normal(loc = 0, scale = 1, size = 20) # 正态分布
        y = rng.normal(loc = 0.5, scale = 1, size = 20)

        t_stats,p_value = Welch()
        welch_t_stats.append(t_stats)
        welch_p_values.append(p_value)

        t_stats,p_value = Wilcoxon()
        wilcoxon_t_stats.append(t_stats)
        wilcoxon_p_values.append(p_value)

    """
    下面开始估测在名义显著性水平0.05下，Welch t检验和Wilcoxon符号秩检验的实际误报率
    """

    welch_false_postives = np.sum(np.array(welch_p_values)<0.05)/1000
    wilcoxon_false_postives = np.sum(np.array(wilcoxon_p_values)<0.05)/1000

    print("Welch t检验的实际误报率:", welch_false_postives)
    print("Wilcoxon符号秩检验的实际误报率:", wilcoxon_false_postives)

#下面开始就这些内容画图：
# Y 组的 loc 依次设为 0, 0.25, 0.5, 0.75, 1.0，每档模拟 1,000 轮。
# 横轴画总体均值差，纵轴画拒绝比例，两种检验各画一条线。
# 横轴为 0 的点对应误报率，其余点对应不同均值差下的功效。

import matplotlib.pyplot as plt

plt.figure(figsize = (10,6)) # 设置画布大小

z = rng.normal(loc = i, scale = 1, size = 20)

for i in [0, 0.25, 0.5, 0.75, 1.0]:
    welch_false_postives = []
    wilcoxon_false_postives = []
    for j in range(1000):
        x = rng.normal(loc = 0, scale = 1, size = 20) # 正态分布
        z = rng.normal(loc = i, scale = 1, size = 20)

        t_stats,p_value = Welch()
        welch_false_postives.append(p_value<0.05)

        t_stats,p_value = Wilcoxon()
        wilcoxon_false_postives.append(p_value<0.05)

    plt.plot(i, np.mean(welch_false_postives), 'ro', label='Welch t检验' if i==0 else "") # 利用推导式画图
    plt.plot(i, np.mean(wilcoxon_false_postives), 'bo', label='Wilcoxon符号秩检验' if i==0 else "")



