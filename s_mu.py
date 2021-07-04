import numpy as np
from pylab import *
from scipy import stats

class Mean(object):
    def __init__(self):
        self.road_length = 200
        self.car_length = 5
        self.max_speed = 75 * 0.277777778

    def time(self):
        d_sigma = 0.25

        s_mu = 0.5
        s_sigma = 0.25
        d_mu = 2.63
        near_dis_plus = 1
        far_dis = 33  #36.6
        target_dis = 19

        # s_mu = 1.5
        # s_sigma = 0.25
        # d_mu = 2.68
        # near_dis_plus = 2
        # far_dis = 33   #36.7
        # target_dis = 20


        # s_mu = 2
        # s_sigma = 0.25
        # d_mu = 2.76
        # near_dis_plus = 3
        # far_dis = 35  # 41
        # target_dis = 21


        # s_mu = 3
        # s_sigma = 0.25
        # d_mu = 3.08
        # near_dis_plus = 5
        # far_dis = 42  # 52.33
        # target_dis = 35

        # s_mu = 5
        # s_sigma = 0.25
        # d_mu = 3.65
        # near_dis_plus = 6
        # far_dis = 75  #  108
        # target_dis = 55

        s_point1 = 20 * 0.277777778
        s_point2 = 40 * 0.277777778
        safe_dis_fig = 10

        data = self.log_zhengtai_mean(s_mu, s_sigma, s_point1, s_point2)
        mean = np.mean(data)
        min_speed = np.min(data)
        safe_dis = min_speed / 0.277777778 - safe_dis_fig  # 安全距离
        near_dis = safe_dis + near_dis_plus
        km = self.road_length / (self.car_length + safe_dis)  # 由安全距离计算最大车辆密度
        k = km / math.exp(mean / self.max_speed)
        # distance = self.road_length / k - self.car_length
        d_point1 = safe_dis
        d_point2 = 100000

        data_dis = self.log_zhengtai_mean(d_mu, d_sigma, d_point1, d_point2)

        return s_mu, s_sigma, d_mu, d_sigma, s_point1, s_point2, d_point1, d_point2, safe_dis, near_dis, far_dis, target_dis, mean, int(math.ceil(k))
        # return s_mu, s_sigma, d_mu, d_sigma, s_point1, s_point2, d_point1, d_point2, safe_dis, near_dis, far_dis, target_dis, mean, 2

        # return d_point1, d_point2

    def log_zhengtai_mean(self, mu, sigma, log_lower, log_upper, data_num=10000):
        norm_lower = np.log(log_lower)
        norm_upper = np.log(log_upper)
        X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
        norm_data = X.rvs(data_num)
        log_data = np.exp(norm_data)
        return log_data

# if __name__ == '__main__':
#     m = Mean()
#     d_sigma = 0.25
#
#     d_mu1 = 2.63
#     d_mu2 = 2.68
#     d_mu3 = 2.76
#     d_mu4 = 3.08
#     d_mu5 = 3.65
#
#
#     a ,b =m.time()
#
#     data1 = m.log_zhengtai_mean(d_mu1,0.25,a,b)
#     print(np.mean(data1))
#     print('max', np.max(data1))
#     plt.hist(data1, density=True, bins=100, alpha=0.7)
#     print(' ')
#
#     data2 = m.log_zhengtai_mean(d_mu2, 0.25, a, b)
#     print(np.mean(data2))
#     print('max', np.max(data2))
#     plt.hist(data2, density=True, bins=100, alpha=0.7)
#     print(' ')
#
#     data3 = m.log_zhengtai_mean(d_mu3, 0.25, a, b)
#     print(np.mean(data3))
#     print('max', np.max(data3))
#     plt.hist(data3, density=True, bins=100, alpha=0.7)
#     print(' ')
#
#     data4 = m.log_zhengtai_mean(d_mu4, 0.25, a, b)
#     print(np.mean(data4))
#     print('max', np.max(data4))
#     plt.hist(data4, density=True, bins=100, alpha=0.7)
#     print(' ')
#
#     data5 = m.log_zhengtai_mean(d_mu5, 0.25, a, b)
#     print(np.mean(data5))
#     print('max', np.max(data5))
#     plt.hist(data5, density=True, bins=100, alpha=0.7)
#     print(' ')
#
#     plt.show()
#
