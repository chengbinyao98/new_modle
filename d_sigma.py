import numpy as np
from pylab import *
from scipy import stats

class Mean(object):
    def __init__(self):
        self.road_length = 200
        self.car_length =5
        self.max_speed = 105 * 0.277777778

    def time1(self):

        s_mu = 0.75
        s_sigma = 0.25
        d_mu = 1.545
        d_sigma = 0.05


        s_mu = 0.75
        s_sigma = 0.25
        d_mu = 1.525
        d_sigma = 0.1


        s_mu = 0.75
        s_sigma = 0.25
        d_mu = 1.235
        d_sigma = 0.25



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

        s_mu = 2
        s_sigma = 0.25
        d_mu = 2.735
        d_sigma = 0.05

        s_mu = 2
        s_sigma = 0.25
        d_mu = 2.73
        d_sigma = 0.1

        s_mu = 2
        s_sigma = 0.25
        d_mu = 2.68
        d_sigma = 0.25



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

        s_mu = 2.5
        s_sigma = 0.25
        d_mu = 3.687
        d_sigma = 0.05

        s_mu = 2.5
        s_sigma = 0.25
        d_mu = 3.683
        d_sigma = 0.1

        s_mu = 2.5
        s_sigma = 0.25
        d_mu = 3.59
        d_sigma = 0.25

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
#     a = 30
#     b = 60
#
#     d_mu1 = 3.687
#     d_mu2 = 3.683
#     d_mu3 = 3.59
#
#     d_sigma1 = 0.05
#     d_sigma2 = 0.1
#     d_sigma3 = 0.25
#
#
#
#     m.time1()
#
#     data = m.log_zhengtai_mean(d_mu1,d_sigma1, a,b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     print(' ')
#     m.time1()
#
#     data = m.log_zhengtai_mean(d_mu2, d_sigma2,a, b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     print(' ')
#     m.time1()
#
#     data = m.log_zhengtai_mean(d_mu3, d_sigma3,a, b)
#     print(np.mean(data))
#     plt.hist(data, density=True, bins=100, alpha=0.7)
#
#     # print(' ')
#     # m.time1()
#     #
#     # data = m.log_zhengtai_mean(d_mu4, d_sigma4,a, b)
#     # print(np.mean(data))
#     # plt.hist(data, density=True, bins=100, alpha=0.7)
#
#
#
#     plt.show()

