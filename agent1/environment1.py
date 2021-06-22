import numpy as np
import math
from pylab import *
from scipy import stats
from d_sigma import Mean
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm


class Env1(object):
    def __init__(self, option):
        m = Mean(option)
        # 固定量
        # 帧结构
        self.frame_slot = 0.01          # 帧时隙时间长度
        self.beam_slot = 100            # 波束选择时隙数
        self.right = 5                  # 正确传输最低的SNR
        # 车辆和道路
        self.road_length = 200          # 道路长度
        self.straight = 100             # 基站和道路的直线距离
        # self.car_length = 5
        # self.max_speed = 105 * 0.277777778            # km/h

        # 存储单元
        self.cars_posit = 0  # 车辆的位置
        self.cars_speed = 0  # 车辆的速度

        # 变化量
        # 算法变化量
        self.road_section = 2            # 每几米划分成一个路段
        self.action_section = 1          # 每几米划分成一个路段
        self.road_range = 35             # 动作可以选择的范围

        # 通信变化量
        self.ann_num = 32                # 天线数目

        # 道路变化量
        self.s_mu, self.s_sigma, self.d_mu, self.d_sigma, self.s_point1, self.s_point2, self.d_point1, self.d_point2 = m.time1()

        # # 车速分布
        # self.d_sigma = 2                                                        # 车辆间距分布
        #
        # # 同一个时段不用变化
        # self.s_point1 = 0 * 0.277777778                                         # 车速范围
        # self.s_point2 = 20 * 0.277777778
        # self.d_point1 = 4                                                       # 车间距范围
        # self.d_point2 = 10
        #
        # safe_dis = self.d_point1                                                # 安全距离
        # km = self.road_length / (self.car_length + safe_dis)                    # 由安全距离计算最大车辆密度
        #
        # data = self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)
        # mean = np.mean(data)
        # k = km / math.exp(mean / self.max_speed)
        # distance = self.road_length / k - self.car_length
        # self.d_mu = math.log(distance) - self.d_sigma * self.d_sigma / 2

    # 由道路上的所有车辆得到所有车辆的路段
    def get_section(self, pos):
        section = math.ceil(pos / self.road_section)
        return section

    def log_zhengtai(self, mu, sigma, log_lower, log_upper, data_num=1):
        norm_lower = np.log(log_lower)
        norm_upper = np.log(log_upper)
        X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
        norm_data = X.rvs(data_num)
        log_data = np.exp(norm_data)
        return log_data


    # def exp(self, scale, low, high, data_num=1):  # scale是均值不是lamda，是1/lamda
    #     rnd_cdf = np.random.uniform(stats.expon.cdf(x=low, scale=scale),
    #                                 stats.expon.cdf(x=high, scale=scale),
    #                                 size=data_num)
    #     return stats.expon.ppf(q=rnd_cdf, scale=scale)

    def get_reward(self, act, reward):
        # 直角边
        a = abs(self.road_length / 2 - self.cars_posit)
        # 斜边
        b = np.sqrt(np.square(a) + np.square(self.straight))
        if self.cars_posit > self.road_length / 2:
            th1 = math.pi - math.acos(a / b)
        else:
            th1 = math.acos(a / b)

        channel = []
        for t in range(self.ann_num):
            m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
            channel.append(m.conjugate())

        # 直角边
        c = abs(self.road_length / 2 - act)
        # 斜边
        d = np.sqrt(np.square(c) + np.square(self.straight))
        if act > self.road_length / 2:
            th2 = math.pi - math.acos(c / d)
        else:
            th2 = math.acos(c / d)

        signal = []
        for t in range(self.ann_num):
            n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
            signal.append(n)

        SNR = np.square(np.linalg.norm(np.dot(channel, signal)))

        if SNR >= self.right:
            reward += 1
        return reward

    def reset(self):
        # 道路环境初始化
        self.cars_speed = self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)[0]
        self.cars_posit = self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]
        # 形成状态
        a = self.get_section(self.cars_posit)
        state = [a,a]
        return state

    def step(self, action, state):

        # 道路的（位置更新）
        reward = 0
        for i in range(self.beam_slot):  # 标记当前车辆是否之前被操作过，保证一个时隙车只跑一个时隙的量
            self.cars_posit += self.cars_speed * self.frame_slot
            reward = self.get_reward(action,reward)

        state_ =[self.get_section(self.cars_posit), state[0]]

        if self.cars_posit > self.road_length:
            done = 1
        else:
            done = 0

        return state_, reward, done

