
import numpy as np
from pylab import *
from scipy import stats

class Mean(object):
    def __init__(self):
        self.road_length = 200
        self.car_length = 5
        self.max_speed = 75 * 0.277777778

    def time(self):

        s_mu = 2
        s_sigma = 0.02
        d_mu = 3.13
        d_sigma = 0.05
        near_dis_plus = 1
        far_dis = 28
        target_dis = 25

        # s_mu = 2
        # s_sigma = 0.02
        # d_mu = 3.13
        # d_sigma = 0.1
        # near_dis_plus = 1
        # far_dis = 32
        # target_dis = 28


        # s_mu = 2
        # s_sigma = 0.02
        # d_mu = 3.13
        # d_sigma = 0.25
        # near_dis_plus = 0.5
        # far_dis = 57
        # target_dis = 38





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

        return s_mu, s_sigma, d_mu, d_sigma, s_point1, s_point2, d_point1, d_point2, safe_dis, near_dis, far_dis, target_dis, mean, int \
            (math.ceil(k))

        # return d_point1, d_point2

    def log_zhengtai_mean(self, mu, sigma, log_lower, log_upper, data_num =10000):
        norm_lower = np.log(log_lower)
        norm_upper = np.log(log_upper)
        X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc =mu, scale =sigma)
        norm_data = X.rvs(data_num)
        log_data = np.exp(norm_data)
        return log_data


if __name__ == '__main__':
    m = Mean()


    d_mu1 = 3.135
    d_mu2 = 3.13
    d_mu3 = 3.125
    d_sigma1 = 0.05
    d_sigma2 = 0.1
    d_sigma3 = 0.25

    b = 100000

    s_mu = 2
    s_sigma1 = 0.02
    s_sigma2 = 0.02
    s_sigma3 = 0.02


    da1 = m.log_zhengtai_mean(s_mu, s_sigma1, 20 * 0.277777778, 40 * 0.277777778)
    mean1 = np.mean(da1)
    min_speed1 = np.min(da1)
    safe_dis1 = min_speed1 / 0.277777778 - 10  # 安全距离
    km1 = m.road_length / (m.car_length + safe_dis1)  # 由安全距离计算最大车辆密度
    k1 = km1 / math.exp(mean1 / m.max_speed)
    distance1 = m.road_length / k1 - m.car_length

    da2 = m.log_zhengtai_mean(s_mu, s_sigma2, 20 * 0.277777778, 40 * 0.277777778)
    mean2 = np.mean(da2)
    min_speed2 = np.min(da2)
    safe_dis2 = min_speed2 / 0.277777778 - 10  # 安全距离
    km2 = m.road_length / (m.car_length + safe_dis2)  # 由安全距离计算最大车辆密度
    k2 = km2 / math.exp(mean2 / m.max_speed)
    distance2 = m.road_length / k2 - m.car_length

    da3 = m.log_zhengtai_mean(s_mu, s_sigma3, 20 * 0.277777778, 40 * 0.277777778)
    mean3 = np.mean(da3)
    min_speed3 = np.min(da3)
    safe_dis3 = min_speed3 / 0.277777778 - 10  # 安全距离
    # distance = self.road_length / k - self.car_length
    km3 = m.road_length / (m.car_length + safe_dis3)  # 由安全距离计算最大车辆密度
    k3 = km3 / math.exp(mean3 / m.max_speed)
    distance3 = m.road_length / k3 - m.car_length




    data1 = m.log_zhengtai_mean(d_mu1 ,d_sigma1 ,safe_dis1 ,b)
    print(distance1, np.mean(data1))
    print('max', np.max(data1))
    plt.hist(data1, density =True, bins =100, alpha =0.7)
    print(' ')

    data2 = m.log_zhengtai_mean(d_mu2, d_sigma2, safe_dis2, b)
    print(distance2, np.mean(data2))
    print('max', np.max(data2))
    plt.hist(data2, density =True, bins =100, alpha =0.7)
    print(' ')

    data3 = m.log_zhengtai_mean(d_mu3, d_sigma3, safe_dis3, b)
    print(distance3, np.mean(data3))
    print('max', np.max(data3))
    plt.hist(data3, density =True, bins =100, alpha =0.7)
    print(' ')


    print(np.min(data1), np.mean(data1), np.max(data1))
    print(np.min(data2), np.mean(data2), np.max(data2))
    print(np.min(data3), np.mean(data3), np.max(data3))



    plt.show()

