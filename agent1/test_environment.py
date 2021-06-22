import numpy as np
import math
from pylab import *
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm


class Env1(object):
    def __init__(self):
        # 固定量
        # 帧结构
        self.frame_slot = 0.01          # 帧时隙时间长度
        self.beam_slot = 100            # 波束选择时隙数
        self.right = 5                  # 正确传输最低的SNR
        # 车辆和道路
        self.road_length = 200          # 道路长度
        self.straight = 100             # 基站和道路的直线距离
        self.car_length = 5
        self.max_speed = 105 * 0.277777778            # km/h

        # 存储单元
        self.cars_posit = []  # 车辆的位置
        self.cars_speed = []  # 车辆的速度

        # 变化量
        # 算法变化量
        self.road_section = 2            # 每几米划分成一个路段
        self.action_section = 1          # 每几米划分成一个路段
        self.road_range = 35             # 动作可以选择的范围

        # 通信变化量
        self.ann_num = 32                # 天线数目

        # 道路变化量
        self.s_mu, self.s_sigma = 0.75, 0.25                                     # 车速分布
        self.d_sigma = 2                                                     # 车辆间距分布

        # 同一个时段不用变化
        self.s_point1 = 0 * 0.277777778                                                       # 车速范围
        self.s_point2 = 20 * 0.277777778
        self.d_point1 = 4                                                       # 车间距范围
        self.d_point2 = 10
        safe_dis = 4                                                            # 安全距离
        km = self.road_length / (self.car_length + safe_dis)                    # 由安全距离计算最大车辆密度
        mean = math.exp(self.s_mu + self.s_sigma * self.s_sigma / 2)            # 由交通流理论计算车辆间距
        k = km / math.exp(mean / self.max_speed)
        distance = self.road_length / k - self.car_length
        self.d_mu = math.log(distance) - self.d_sigma * self.d_sigma / 2

    # 由道路上的所有车辆得到所有车辆的路段
    def get_section(self, list):
        section = []
        for i in range(len(list)):
            section.append(math.ceil(list[i] / self.road_section))
        return section

    def log_zhengtai(self, mu, sigma, log_lower, log_upper, data_num=1):
        norm_lower = np.log(log_lower)
        norm_upper = np.log(log_upper)
        X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
        norm_data = X.rvs(data_num)
        log_data = np.exp(norm_data)
        return log_data

    def exp(self, scale, low, high, data_num=1):  # scale是均值不是lamda，是1/lamda
        rnd_cdf = np.random.uniform(stats.expon.cdf(x=low, scale=scale),
                                    stats.expon.cdf(x=high, scale=scale),
                                    size=data_num)
        return stats.expon.ppf(q=rnd_cdf, scale=scale)

    def get_information(self, section):
        for i in range(10):  # 这个10随便，只要保证能新加上所有的车辆即可
            # 生成一个新的车辆进入，初始化车辆间距
            dis1 = self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]
            dis2 = self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]
            if self.cars_posit[0] >= dis1 + dis2:
                section.insert(0, (self.cars_posit[0] - dis1) / self.road_section)
                self.cars_posit.insert(0, (self.cars_posit[0] - dis1))  # 车辆的位置（位置更新）
                self.cars_speed.insert(0, self.cars_speed[0])  # 车辆的速度（位置更新）
            else:
                break
        for i in range(10):
            # 将超出道路的车辆排除
            if self.cars_posit[len(self.cars_posit) - 1] > self.road_length:
                del section[len(self.cars_posit) - 1]
                del self.cars_speed[len(self.cars_posit) - 1]
                del self.cars_posit[len(self.cars_posit) - 1]
            else:
                break
        return section

    def get_reward(self, act, reward):
        for j in range(len(act)):
            # 直角边
            a = abs(self.road_length / 2 - self.cars_posit[j])
            # 斜边
            b = np.sqrt(np.square(a) + np.square(self.straight))
            if self.cars_posit[j] > self.road_length / 2:
                th1 = math.pi - math.acos(a / b)
            else:
                th1 = math.acos(a / b)

            channel = []
            for t in range(self.ann_num):
                m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
                channel.append(m.conjugate())

            # 直角边
            c = abs(self.road_length / 2 - act[j])
            # 斜边
            d = np.sqrt(np.square(c) + np.square(self.straight))
            if act[j] > self.road_length / 2:
                th2 = math.pi - math.acos(c / d)
            else:
                th2 = math.acos(c / d)

            signal = []
            for t in range(self.ann_num):
                n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
                signal.append(n)

            SNR = np.square(np.linalg.norm(np.dot(channel, signal)))

            if SNR >= self.right:
                reward[j] += 1
        return reward

    def reset(self):
        self.cars_posit = []
        self.cars_speed = []
        speed = self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)[0]
        for i in range(50):  # 任意数目都可以，主要是用于生成路段上的车辆
            dis = self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]
            # 生成车辆的初始位置和速度
            if i == 0:
                self.cars_posit.append(dis)
                self.cars_speed.append(speed)
            else:
                y = self.cars_posit[i - 1] + dis
                if y >= self.road_length:
                    break
                else:
                    self.cars_posit.append(y)
                    self.cars_speed.append(speed)
        # 形成状态
        a = self.get_section(self.cars_posit)
        state = []
        for i in range(len(a)):
            b = [a[i], a[i]]
            state.append(b)
        return state

    def step(self, action, state):
        state0 = []
        for i in range(len(state)):
            state0.append(state[i][0])

        # 道路的（位置更新）
        reward = [0 for i in range(len(action))]
        for i in range(self.beam_slot):  # 标记当前车辆是否之前被操作过，保证一个时隙车只跑一个时隙的量
            for j in range(len(action)):
                self.cars_posit[j] += self.cars_speed[j] * self.frame_slot
            reward = self.get_reward(action,reward)

        state1 = self.get_section(self.cars_posit)

        draw_state_ = []
        for i in range(len(state0)):
            draw_state_.append([state1[i],state0[i]])


        section = self.get_information(state0)
        state_ =[]
        section_ = self.get_section(self.cars_posit)
        for i in range(len(self.cars_posit)):
            state_.append([section_[i], section[i]])

        return state_, reward, draw_state_

