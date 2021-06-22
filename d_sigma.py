import numpy as np
from pylab import *
from scipy import stats

class Mean(object):
    def __init__(self):
        self.road_length = 200
        self.car_length =5
        self.max_speed = 105 * 0.277777778

    def time1(self):
        d_sigma = 0.25

        s_mu = -1
        s_sigma = 0.25
        d_mu = 0.01

        # s_mu = 0
        # s_sigma = 0.25
        # d_mu = 0.72
        #
        # s_mu = 0.75
        # s_sigma = 0.25
        # d_mu = 1.22
        #
        # s_mu = 1.5
        # s_sigma = 0.25
        # d_mu = 1.55
        #
        # s_mu = 2.5
        # s_sigma = 0.25
        # d_mu = 1.68

        s_point1 = 0 * 0.277777778
        s_point2 = 20 * 0.277777778
        d_point1 = 4
        d_point2 = 10

        safe_dis = d_point1  # 安全距离
        km = self.road_length / (self.car_length + safe_dis)  # 由安全距离计算最大车辆密度

        data = self.log_zhengtai_mean(s_mu, s_sigma, s_point1, s_point2)
        mean = np.mean(data)

        k = km / math.exp(mean / self.max_speed)
        distance = self.road_length / k - self.car_length

        return s_mu, s_sigma, d_mu, d_sigma, s_point1, s_point2, d_point1, d_point2


    def time2(self):
        d_sigma = 0.25

        s_mu = 0.5
        s_sigma = 0.25
        d_mu = 2.51

        # s_mu = 1.5
        # s_sigma = 0.25
        # d_mu = 2.57
        #
        # s_mu = 2
        # s_sigma = 0.25
        # d_mu = 2.67
        #
        # s_mu = 3
        # s_sigma = 0.25
        # d_mu = 2.88
        #
        # s_mu = 5
        # s_sigma = 0.25
        # d_mu = 2.92

        s_point1 = 20 * 0.277777778
        s_point2 = 40 * 0.277777778
        d_point1 = 10
        d_point2 = 30

        safe_dis = d_point1  # 安全距离
        km = self.road_length / (self.car_length + safe_dis)  # 由安全距离计算最大车辆密度

        data = self.log_zhengtai_mean(s_mu, s_sigma, s_point1, s_point2)
        mean = np.mean(data)

        k = km * (1 - mean / self.max_speed)
        distance = self.road_length / k - self.car_length

        return s_mu, s_sigma, d_mu, d_sigma, s_point1, s_point2, d_point1, d_point2

    def time3(self):
        d_sigma = 0.25

        s_mu = 0.5
        s_sigma = 0.25
        d_mu = 2.75

        # s_mu = 1.5
        # s_sigma = 0.25
        # d_mu = 3.05
        #
        # s_mu = 2.5
        # s_sigma = 0.25
        # d_mu = 3.59
        #
        # s_mu = 4
        # s_sigma = 0.25
        # d_mu = 4.31
        #
        # s_mu = 6
        # s_sigma = 0.25
        # d_mu = 4.53

        s_point1 = 40 * 0.277777778
        s_point2 = 60 * 0.277777778
        d_point1 = 30
        d_point2 = 60

        safe_dis = d_point1  # 安全距离
        km = self.road_length / (self.car_length + safe_dis)  # 由安全距离计算最大车辆密度

        data = self.log_zhengtai_mean(s_mu, s_sigma, s_point1, s_point2)
        mean = np.mean(data)

        k = km * (-1) *math.log(mean/self.max_speed)
        distance = self.road_length / k - self.car_length

        return s_mu, s_sigma, d_mu, d_sigma, s_point1, s_point2, d_point1, d_point2




    def log_zhengtai_mean(self, mu, sigma, log_lower, log_upper, data_num=10000):
        norm_lower = np.log(log_lower)
        norm_upper = np.log(log_upper)
        X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
        norm_data = X.rvs(data_num)
        log_data = np.exp(norm_data)
        return log_data



#
# if __name__ == '__main__':
#     m = Mean()
#     d_sigma = 0.25
#     a = 60
#     b = 100
#
#     d_mu1 = 2.75
#     d_mu2 = 3.05
#     d_mu3 = 3.59
#     d_mu4 = 4.31
#     d_mu5 = 4.53
#
#
#     m.time1(0.5, 0.25)
#
#     data = m.exp(d_mu1,a,b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     print(' ')
#     m.time1(1.5, 0.25)
#
#     data = m.exp(d_mu2, a, b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     print(' ')
#     m.time1(2.5, 0.25)
#
#     data = m.exp(d_mu3, a, b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     print(' ')
#     m.time1(4, 0.25)
#
#     data = m.exp(d_mu4, a, b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     print(' ')
#     m.time1(6, 0.25)
#
#     data = m.exp(d_mu5, a, b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     plt.show()

