import numpy as np
import math
from pylab import *
from scipy import stats


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.stats import norm


class Env2(object):
    def __init__(self):
        from cluster_runner import m
        # 固定量
        # 帧结构
        self.frame_slot = 0.01  # 帧时隙时间长度
        self.beam_slot = 100  # 波束选择时隙数
        self.right = 5  # 正确传输最低的SNR
        # 车辆和道路
        self.road_length = 200  # 道路长度
        self.straight = 100  # 基站和道路的直线距离

        # 存储单元
        self.cars_posit = []  # 车辆的位置（连续）
        self.cars_speed = []  # 车辆的速度（连续

        # 变化量
        # 算法变化量
        self.road_section = 2  # 每几米划分成一个路段
        self.action_section = 2  # 每几米划分成一个路段
        self.road_range = 35  # 动作可以选择的范围

        # 通信变化量
        self.ann_num = 32  # 天线数目
        self.no_interference = 30  # 随着天线个数变化
        self.s_mu, self.s_sigma, self.d_mu, self.d_sigma, self.s_point1, self.s_point2, self.d_point1, self.d_point2, \
        self.safe_dis, self.near_dis, self.far_dis, self.target_dis, \
        self.mean_speed, \
        self.car_num = m.time()

        self.batch_size = self.car_num - 1
        # self.seq_batch = []
        # self.res_batch = []

        # self.no_num_change = 3

    # region 【功能函数】生成截断对数正态分布，要求对数正态在[log_lower,log_upper]
    def log_zhengtai(self, mu, sigma, log_lower, log_upper, data_num = 1):
        norm_lower = np.log(log_lower)
        norm_upper = np.log(log_upper)
        X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc = mu, scale = sigma)
        norm_data = X.rvs(data_num)
        log_data = np.exp(norm_data)
        return log_data

    def road_step(self):
        count = 1 / 2
        for i in range(len(self.cars_posit)):
            if i != len(self.cars_posit) - 1:
                if self.cars_posit[i + 1] - self.cars_posit[i] <= self.near_dis:
                    self.cars_posit[i] = self.cars_posit[i + 1] - self.safe_dis - (
                                self.cars_posit[i + 1] - self.safe_dis - self.cars_posit[i]) / 2
                    self.cars_speed[i] = self.cars_speed[i + 1]
                if self.cars_posit[i + 1] - self.cars_posit[i] >= self.far_dis:
                    self.cars_posit[i] = self.cars_posit[i + 1] - self.target_dis - (
                                self.cars_posit[i + 1] - self.target_dis - self.cars_posit[i]) / 2
                    self.cars_speed[i] = self.mean_speed + np.random.uniform(-0.5, 0.5)
                if self.near_dis < self.cars_posit[i + 1] - self.cars_posit[i] < self.far_dis:
                    self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
            else:
                self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]
                # self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.frame_slot * self.frame_slot * add / 2 + self.cars_posit[i]
                # self.cars_speed[i] = min(self.cars_speed[i] + add * self.frame_slot, self.max_speed)

    def get_reward(self, act,pos,reward,n):
        if pos[1] - pos[0] <= self.no_interference:
            for i in range(n):
                SNR_noise = 0
                SNR = 0
                for j in range(n):
                    # 直角边
                    a = abs(self.road_length / 2 - pos[i])
                    # 斜边
                    b = np.sqrt(np.square(a) + np.square(self.straight))
                    if pos[i] > self.road_length / 2:
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
                        p = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
                        signal.append(p)

                    if j != i:
                        SNR_noise += np.square(np.linalg.norm(np.dot(channel, signal)))
                    else:
                        SNR = np.square(np.linalg.norm(np.dot(channel, signal)))
                if SNR_noise == 0:
                    if SNR >= self.right:
                        reward[i] += 1
                else:
                    if SNR / SNR_noise >= self.right:
                        reward[i] += 1

        else:
            for i in range(n):
                a = abs(self.road_length / 2 - pos[i])
                # 斜边
                b = np.sqrt(np.square(a) + np.square(self.straight))
                if pos[i] > self.road_length / 2:
                    th1 = math.pi - math.acos(a / b)
                else:
                    th1 = math.acos(a / b)

                channel = []
                for t in range(self.ann_num):
                    m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
                    channel.append(m.conjugate())

                # 直角边
                c = abs(self.road_length / 2 - act[i])
                # 斜边
                d = np.sqrt(np.square(c) + np.square(self.straight))
                if act[i] > self.road_length / 2:
                    th2 = math.pi - math.acos(c / d)
                else:
                    th2 = math.acos(c / d)

                signal = []
                for t in range(self.ann_num):
                    n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
                    signal.append(n)

                SNR = np.square(np.linalg.norm(np.dot(channel, signal)))

                if SNR >= self.right:
                    reward[i] += 1
        return reward

    def reset(self,n):
        self.cars_posit = []
        self.cars_speed = []
        for i in range(2):  # 任意数目都可以，主要是用于生成路段上的车辆
            speed = self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)[0]
            dis = self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]
            # 生成车辆的初始位置和速度
            if i == 0:
                self.cars_posit.append(dis)
                self.cars_speed.append(speed)
            else:
                y = self.cars_posit[i - 1] + dis
                self.cars_posit.append(y)
                self.cars_speed.append(speed)
        # print(self.cars_speed)
        # print(self.cars_posit)

        a = [self.cars_posit[0],self.cars_posit[1]]

        fake = []
        for i in range(self.beam_slot):
            self.road_step()
            fake.append([self.cars_posit[0],self.cars_posit[1]])

        state = [a[0], a[1], self.cars_posit[0], self.cars_posit[1]]

        return state, fake

    def step(self, action, fake, n):
        # 道路的（位置更新）
        reward = [0 for p in range(n)]
        fake2 = []
        for i in range(self.beam_slot):  # 标记当前车辆是否之前被操作过，保证一个时隙车只跑一个时隙的量
            reward = self.get_reward(action, fake[i], reward, 2)
            self.road_step()
            fake2.append([self.cars_posit[0],self.cars_posit[1]])

        state_ = [fake[self.beam_slot - 1][0], fake[self.beam_slot - 1][1], self.cars_posit[0], self.cars_posit[1]]

        if fake[self.beam_slot - 1][1] > self.road_length:
            done = 1
        else:
            done = 0

        total_reward = 0
        for i in range(len(reward)):
            total_reward += reward[i]

        return state_, total_reward, done, fake2



