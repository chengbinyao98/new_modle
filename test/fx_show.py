# import numpy as np
# from pylab import *
# from scipy import stats
# import matplotlib
# import matplotlib.pyplot as plt
#
# # 设置matplotlib正常显示中文和负号
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
# matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
#
# road_length = 200
# car_length = 5
# max_speed = 105*0.277777778
#
# # region 【功能函数】生成截断对数正态分布，要求对数正态在[log_lower,log_upper]
# def get_trunc_lognorm(mu, sigma, log_lower, log_upper, data_num=10000):
#     norm_lower = np.log(log_lower)
#     norm_upper = np.log(log_upper)
#     X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
#     norm_data = X.rvs(data_num)
#     log_data = np.exp(norm_data)
#     return log_data
#
# def exp(scale, low, high, data_num = 10000):  # scale是均值不是lamda，是1/lamda
#     rnd_cdf = np.random.uniform(stats.expon.cdf(x = low, scale = scale),
#                                 stats.expon.cdf(x = high, scale = scale),
#                                 size = data_num)
#     return stats.expon.ppf(q = rnd_cdf, scale = scale)
#
# s_mu, s_sigma = 0, 1
# s_point1 = 0
# s_point2 = 20*0.277777778
# log_data = get_trunc_lognorm(s_mu, s_sigma, s_point1, s_point2)
#
# # safe_dis = 4  # 安全距离
# # km = road_length / (car_length + safe_dis)  # 由安全距离计算最大车辆密度
# # mean = math.exp(s_mu + s_sigma * s_sigma / 2)  # 由交通流理论计算车辆间距
# # k = km / math.exp(mean / max_speed)
# # distance = road_length / k - car_length
# # d_mu = math.log(distance) - d_sigma * d_sigma / 2
#
# d_lamda = 2
# d_point1 = 0
# d_point2 = 100000000
# exp_data = exp(d_lamda, d_point1, d_point2)
#
#
# # plt.hist(log_data, stacked=True, bins=100)
# plt.hist(exp_data, stacked=True, bins=100)
# # plt.title("所求的截断对数正态分布")
# plt.show()

import numpy as np
from pylab import *
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm

road_length = 200
max_speed = 105  * 0.277777778
car_length = 5


# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# region 【功能函数】生成截断对数正态分布，要求对数正态在[log_lower,log_upper]
def get_trunc_lognorm(mu, sigma, log_lower, log_upper=np.inf, data_num=10000):
    norm_lower = np.log(log_lower)
    norm_upper = np.log(log_upper)
    X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
    norm_data = X.rvs(data_num)
    log_data = np.exp(norm_data)
    return log_data

def trunc_exp_rv(scale, low, high, data_num=10000): # scale是均值不是lamda，是1/lamda
    rnd_cdf = np.random.uniform(stats.expon.cdf(x=low, scale=scale),
                                stats.expon.cdf(x=high, scale=scale),
                                size=data_num)
    print(rnd_cdf)
    return stats.expon.ppf(q=rnd_cdf, scale=scale)



s_point1 = 60 * 0.277777778
s_point2 = 80 * 0.277777778

d_point1 = 60
d_point2 = 100

safe_dis = 60  # 安全距离
km = road_length / (car_length + safe_dis)                # 由安全距离计算最大车辆密度



s_mu1 = 0.5
s_mu2 = 2
s_mu3 = 3
s_mu4 = 4.5
s_mu5 = 7

s_sigma1 = 0.01
s_sigma2 = 0.02
s_sigma3 = 0.05
s_sigma4 = 0.25

d_sigma1 = 0.02
d_sigma2 = 0.05
d_sigma3 = 0.1
d_sigma4 = 10

# # s_mu变化图
# s_sigma = s_sigma4
# d_sigma = d_sigma4
#
# log_data1 = get_trunc_lognorm(s_mu1, s_sigma, s_point1, s_point2)
# plt.hist(log_data1, density=True, bins=100,alpha=0.7)
# log_data2 = get_trunc_lognorm(s_mu2, s_sigma, s_point1, s_point2)
# plt.hist(log_data2, density=True, bins=100,alpha=0.7)
# log_data3 = get_trunc_lognorm(s_mu3, s_sigma, s_point1, s_point2)
# plt.hist(log_data3, density=True, bins=100,alpha=0.7)
# log_data4 = get_trunc_lognorm(s_mu4, s_sigma, s_point1, s_point2)
# plt.hist(log_data4, density=True, bins=100,alpha=0.7)
# log_data5 = get_trunc_lognorm(s_mu5, s_sigma, s_point1, s_point2)
# plt.hist(log_data5, density=True, bins=100,alpha=0.7)
# plt.title("speed")
# plt.show()



# s_sigma变化图
s_mu = s_mu3
d_sigma = d_sigma4

log_data1 = get_trunc_lognorm(2.5, 0.25, 40* 0.277777778, 60* 0.277777778)
plt.hist(log_data1, density=True, bins=100,alpha=0.7)
# log_data2 = get_trunc_lognorm(s_mu, s_sigma2, s_point1, s_point2)
# plt.hist(log_data2, density=True, bins=100,alpha=0.7)
# log_data3 = get_trunc_lognorm(s_mu, s_sigma3, s_point1, s_point2)
# plt.hist(log_data3, density=True, bins=100,alpha=0.7)
# log_data4 = get_trunc_lognorm(s_mu, s_sigma4, s_point1, s_point2)
# plt.hist(log_data4, density=True, bins=100,alpha=0.7)
plt.title("speed")
plt.show()







# # s_mu变化引起的d_mu变化，此时s_sigma = s_sigma3 = 0.25。d_sigma = d_sigma4 = 2
# s_sigma = s_sigma4
# d_sigma = d_sigma4
#
# mean1 = math.exp(s_mu1 + s_sigma * s_sigma / 2)        # 由交通流理论计算车辆间距
# k1 = km / math.exp(mean1 / max_speed)
# distance1 = road_length / k1 - car_length
# d_mu1 = math.log(distance1) - d_sigma * d_sigma / 2
#
# mean2 = math.exp(s_mu2 + s_sigma * s_sigma / 2)        # 由交通流理论计算车辆间距
# k2 = km / math.exp(mean2 / max_speed)
# distance2 = road_length / k2 - car_length
# d_mu2 = math.log(distance2) - d_sigma * d_sigma / 2
#
# mean3 = math.exp(s_mu3 + s_sigma * s_sigma / 2)        # 由交通流理论计算车辆间距
# k3 = km / math.exp(mean3 / max_speed)
# distance3 = road_length / k3 - car_length
# d_mu3 = math.log(distance3) - d_sigma * d_sigma / 2
#
# mean4 = math.exp(s_mu4 + s_sigma * s_sigma / 2)        # 由交通流理论计算车辆间距
# k4 = km / math.exp(mean4 / max_speed)
# distance4 = road_length / k4 - car_length
# d_mu4 = math.log(distance4) - d_sigma * d_sigma / 2
#
# mean5 = math.exp(s_mu5 + s_sigma * s_sigma / 2)        # 由交通流理论计算车辆间距
# k5 = km / math.exp(mean5 / max_speed)
# distance5 = road_length / k5 - car_length
# d_mu5 = math.log(distance5) - d_sigma * d_sigma / 2


# # s_sigma变化引起的d_mu变化，此时s_mu = s_mu3 = 0.75。d_sigma = d_sigma4 = 2
# s_mu = s_mu3
# d_sigma = d_sigma4
#
# mean1 = math.exp(s_mu + s_sigma1 * s_sigma1 / 2)        # 由交通流理论计算车辆间距
# k1 = km / math.exp(mean1 / max_speed)
# distance1 = road_length / k1 - car_length
# d_mu1 = math.log(distance1) - d_sigma * d_sigma / 2
#
# mean2 = math.exp(s_mu + s_sigma2 * s_sigma2 / 2)        # 由交通流理论计算车辆间距
# k2 = km / math.exp(mean2 / max_speed)
# distance2 = road_length / k2 - car_length
# d_mu2 = math.log(distance2) - d_sigma * d_sigma / 2
#
# mean3 = math.exp(s_mu + s_sigma3 * s_sigma3 / 2)        # 由交通流理论计算车辆间距
# k3 = km / math.exp(mean3 / max_speed)
# distance3 = road_length / k3 - car_length
# d_mu3 = math.log(distance3) - d_sigma * d_sigma / 2
#
# mean4 = math.exp(s_mu + s_sigma4 * s_sigma4 / 2)        # 由交通流理论计算车辆间距
# k4 = km / math.exp(mean4 / max_speed)
# distance4 = road_length / k4 - car_length
# d_mu4 = math.log(distance4) - d_sigma * d_sigma / 2


# # s变化引起的d_mu变化
# d_sigma = d_sigma4
#
# log_data1 = get_trunc_lognorm(d_mu1, d_sigma, d_point1, d_point2)
# plt.hist(log_data1, density=True, bins=100,alpha=0.7)
# log_data2 = get_trunc_lognorm(d_mu2, d_sigma, d_point1, d_point2)
# plt.hist(log_data2, density=True, bins=100,alpha=0.7)
# log_data3 = get_trunc_lognorm(d_mu3, d_sigma, d_point1, d_point2)
# plt.hist(log_data3, density=True, bins=100,alpha=0.7)
# log_data4 = get_trunc_lognorm(d_mu4, d_sigma, d_point1, d_point2)
# plt.hist(log_data4, density=True, bins=100,alpha=0.7)
# # log_data5 = get_trunc_lognorm(d_mu5, d_sigma, d_point1, d_point2)
# # plt.hist(log_data5, density=True, bins=100,alpha=0.7)
# plt.title("distance")
# plt.show()








# # 由d_sigma变化，此时s_mu = s_mu3 = 0.75，s_sigma = s_sigma3 = 0.25
# s_mu = s_mu3
# s_sigma = s_sigma4
#
# mean1 = math.exp(s_mu + s_sigma * s_sigma / 2)        # 由交通流理论计算车辆间距
# k1 = km / math.exp(mean1 / max_speed)
# distance1 = road_length / k1 - car_length
# d_mu1 = math.log(distance1) - d_sigma1 * d_sigma1 / 2
#
# log_data1 = get_trunc_lognorm(d_mu1, d_sigma1, d_point1, d_point2)
# plt.hist(log_data1, density=True, bins=100,alpha=0.7)
# log_data2 = get_trunc_lognorm(d_mu1, d_sigma2, d_point1, d_point2)
# plt.hist(log_data2, density=True, bins=100,alpha=0.7)
# log_data3 = get_trunc_lognorm(d_mu1, d_sigma3, d_point1, d_point2)
# plt.hist(log_data3, density=True, bins=100,alpha=0.7)
# log_data4 = get_trunc_lognorm(d_mu1, d_sigma4, d_point1, d_point2)
# plt.hist(log_data4, density=True, bins=100,alpha=0.7)
# plt.title("distance")
# plt.show()

#
# # 观测车距
# s_mu = s_mu3
# s_sigma = s_sigma4
#
# mean1 = math.exp(s_mu + s_sigma * s_sigma / 2)        # 由交通流理论计算车辆间距
# k1 = km / math.exp(mean1 / max_speed)
# distance1 = road_length / k1 - car_length
#
# log_data1 = trunc_exp_rv(distance1, d_point1, d_point2)
# plt.hist(log_data1, density=True, bins=100,alpha=0.7)
# log_data2 = trunc_exp_rv(distance1, d_point1, d_point2)
# plt.hist(log_data2, density=True, bins=100,alpha=0.7)
# log_data3 = trunc_exp_rv(distance1, d_point1, d_point2)
# plt.hist(log_data3, density=True, bins=100,alpha=0.7)
# log_data4 = trunc_exp_rv(distance1, d_point1, d_point2)
# plt.hist(log_data4, density=True, bins=100,alpha=0.7)
# log_data5 = trunc_exp_rv(distance1, d_point1, d_point2)
# plt.hist(log_data5, density=True, bins=100,alpha=0.7)
# plt.title("distance")
# plt.show()

