import numpy as np
from pylab import *
from scipy import stats

class Mean(object):
    def __init__(self):
        self.road_length = 200
        self.car_length =5
        self.max_speed = 105 * 0.277777778

    def time1(self, option):
        d_sigma = 0.25
        if option == 1:
            s_mu = 0.75
            s_sigma = 0.05
            d_mu = 1.215
        if option == 2:
            s_mu = 0.75
            s_sigma = 0.1
            d_mu = 1.23
        if option == 3:
            s_mu = 0.75
            s_sigma = 0.25
            d_mu = 1.24
        if option == 4:
            s_mu = 0.75
            s_sigma = 0.8
            d_mu = 1.24


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


    def time2(self, option):
        d_sigma = 0.25
        if option == 1:
            s_mu = 2
            s_sigma = 0.02
            d_mu = 2.649
        if option == 2:
            s_mu = 2
            s_sigma = 0.05
            d_mu = 2.649
        if option == 3:
            s_mu = 2
            s_sigma = 0.1
            d_mu = 2.657
        if option == 4:
            s_mu = 2
            s_sigma = 0.25
            d_mu = 2.678


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

        s_mu = 2.5
        s_sigma = 0.01
        d_mu = 3.237

        # s_mu = 2.5
        # s_sigma = 0.02
        # d_mu = 3.237
        #
        # s_mu = 2.5
        # s_sigma = 0.05
        # d_mu = 3.27
        #
        # s_mu = 2.5
        # s_sigma = 0.25
        # d_mu = 3.59

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




# if __name__ == '__main__':
#     m = Mean()
#     d_sigma = 0.25
#     a = 30
#     b = 60
#
#     d_mu1 = 3.237
#     d_mu2 = 3.237
#     d_mu3 = 3.27
#     d_mu4 = 3.59
#
#
#
#     m.time1(2.5, 0.01)
#
#     data = m.log_zhengtai_mean(d_mu1,d_sigma, a,b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     print(' ')
#     m.time1(2.5, 0.02)
#
#     data = m.log_zhengtai_mean(d_mu2, d_sigma,a, b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     print(' ')
#     m.time1(2.5, 0.05)
#
#     data = m.log_zhengtai_mean(d_mu3, d_sigma,a, b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     print(' ')
#     m.time1(2.5, 0.25)
#
#     data = m.log_zhengtai_mean(d_mu4, d_sigma,a, b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#
#
#     plt.show()

