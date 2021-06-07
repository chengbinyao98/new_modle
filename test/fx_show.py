import numpy as np
from pylab import *
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

road_length = 200
car_length = 5
max_speed = 105*0.277777778

# region 【功能函数】生成截断对数正态分布，要求对数正态在[log_lower,log_upper]
def get_trunc_lognorm(mu, sigma, log_lower, log_upper, data_num=10000):
    norm_lower = np.log(log_lower)
    norm_upper = np.log(log_upper)
    X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
    norm_data = X.rvs(data_num)
    log_data = np.exp(norm_data)
    return log_data

def exp(scale, low, high, data_num = 10000):  # scale是均值不是lamda，是1/lamda
    rnd_cdf = np.random.uniform(stats.expon.cdf(x = low, scale = scale),
                                stats.expon.cdf(x = high, scale = scale),
                                size = data_num)
    return stats.expon.ppf(q = rnd_cdf, scale = scale)

s_mu, s_sigma = 0, 1
s_point1 = 0
s_point2 = 20*0.277777778
log_data = get_trunc_lognorm(s_mu, s_sigma, s_point1, s_point2)

# safe_dis = 4  # 安全距离
# km = road_length / (car_length + safe_dis)  # 由安全距离计算最大车辆密度
# mean = math.exp(s_mu + s_sigma * s_sigma / 2)  # 由交通流理论计算车辆间距
# k = km / math.exp(mean / max_speed)
# distance = road_length / k - car_length
# d_mu = math.log(distance) - d_sigma * d_sigma / 2

d_lamda = 2
d_point1 = 0
d_point2 = 100000000
exp_data = exp(d_lamda, d_point1, d_point2)


# plt.hist(log_data, stacked=True, bins=100)
plt.hist(exp_data, stacked=True, bins=100)
# plt.title("所求的截断对数正态分布")
plt.show()