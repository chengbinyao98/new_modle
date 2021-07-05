import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from pylab import *
from scipy import stats


import matplotlib
import matplotlib.mlab as mlab
from scipy.stats import norm


class Env(object):
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
        self.action_section = 1  # 每几米划分成一个路段
        self.road_range = 20  # 动作可以选择的范围

        # 通信变化量
        self.ann_num = 64  # 天线数目
        self.no_interference = 30  # 随着天线个数变化
        self.s_mu, self.s_sigma, self.d_mu, self.d_sigma, self.s_point1, self.s_point2, self.d_point1, self.d_point2, \
        self.safe_dis, self.near_dis, self.far_dis, self.target_dis, \
        self.mean_speed, \
        self.car_num = m.time()

        self.batch_size = self.car_num - 1
        # self.seq_batch = []
        # self.res_batch = []

        # self.no_num_change = 3


    def log_zhengtai(self, mu, sigma, log_lower, log_upper, data_num = 1):
        norm_lower = np.log(log_lower)
        norm_upper = np.log(log_upper)
        X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc = mu, scale = sigma)
        norm_data = X.rvs(data_num)
        log_data = np.exp(norm_data)
        return log_data

    # 道路初始化
    def road_reset(self):
        self.cars_posit = []
        self.cars_speed = []
        for i in range(self.car_num):  # 任意数目都可以，主要是用于生成路段上的车辆
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

    # 道路路面现有车辆更新
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

    def get_information(self):
        mark = 0
        for i in range(10):  # 这个10随便，只要保证能新加上所有的车辆即可
            # 生成一个新的车辆进入，初始化车辆间距
            dis1 = self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]
            dis2 = self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]
            speed = self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)[0]
            if self.cars_posit[0] >= dis1 + dis2:
                self.cars_posit.insert(0, (self.cars_posit[0] - dis1))  # 车辆的位置（位置更新）
                self.cars_speed.insert(0, speed)  # 车辆的速度（位置更新）
                mark += 1
                # print('mark',mark)
            else:
                break
        for j in range(mark):
            # 将超出道路的车辆排除
            # if self.cars_posit[len(self.cars_posit) - 1] > self.road_length:
            del self.cars_speed[len(self.cars_posit) - 1]
            del self.cars_posit[len(self.cars_posit) - 1]
            # else:
            #     break

    def get_reward(self, list_act, pos, list_reward, tool, info):
        for i in range(len(list_act)):
            SNR_noise = 0
            SNR = 0
            for num in range(i - info[i][2], i - info[i][2] + info[i][0]):
                # 直角边
                a = abs(self.road_length / 2 - pos[num])
                # 斜边
                b = np.sqrt(np.square(a) + np.square(self.straight))
                if pos[num] > self.road_length / 2:
                    th1 = math.pi - math.acos(a / b)
                else:
                    th1 = math.acos(a / b)

                channel = []
                for t in range(self.ann_num):
                    m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
                    channel.append(m.conjugate())

                # 直角边
                c = abs(self.road_length / 2 - list_act[i] )
                # 斜边
                d = np.sqrt(np.square(c) + np.square(self.straight))
                if list_act[i]  > self.road_length / 2:
                    th2 = math.pi - math.acos(c / d)
                else:
                    th2 = math.acos(c / d)

                signal = []
                for t in range(self.ann_num):
                    n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
                    signal.append(n)

                if num != i:
                    SNR_noise += np.square(np.linalg.norm(np.dot(channel, signal)))
                else:
                    SNR = np.square(np.linalg.norm(np.dot(channel, signal)))
            if SNR_noise == 0:
                if SNR >= self.right:
                    list_reward[i] += 1
            else:
                if SNR / SNR_noise >= self.right:
                    list_reward[i] += 1
        return list_reward

    def reset(self, tool):
        self.cars_posit = []
        self.cars_speed = []
        # 道路环境初始化
        self.road_reset()
        # 获得道路上的每个车辆信息
        info = tool.get_info(self.cars_posit, self.no_interference)

        # 形成状态
        tem = []
        for k in range(len(self.cars_posit)):
            tem.append(self.cars_posit[k])
        b = tool.classify(tem, info)
        fake = []
        for i in range(self.beam_slot):
            self.road_step()
            temp = []
            for j in range(len(self.cars_posit)):
                temp.append(self.cars_posit[j])
            fake.append(temp)
        c = tool.classify(self.cars_posit, info)
        d = tool.integrate(b, c)
        return d, fake

    def step(self, dic_action, fake, tool):
        info = tool.get_info(fake[0], self.no_interference)  # 当前道路的信息
        # print(info)
        # print(tool.get_info1(self.cars_posit, self.no_interference))
        action = tool.reverse_classify(dic_action, info)  # 当前车辆的动作

        self.get_information()
        tem = []
        for k in range(len(self.cars_posit)):
            tem.append(self.cars_posit[k])
        next_info = tool.get_info(tem, self.no_interference)  # 下一时刻的道路信息

        # 道路的（位置更新）
        reward = [0 for p in range(len(info))]  # 用于记录一个帧周期的车辆情况
        fake2 = []
        for i in range(self.beam_slot):
            reward = self.get_reward(action, fake[i], reward, tool, info)
            self.road_step()
            temp = []
            for j in range(len(self.cars_posit)):
                temp.append(self.cars_posit[j])
            fake2.append(temp)

        dic_reward = tool.classify(reward, info)



        # 下一时刻的状态（位置更新）（数目更新）
        b = tool.classify(tem, next_info)
        c = tool.classify(self.cars_posit, next_info)
        dic_state_ = tool.integrate(b, c)


        return dic_state_, dic_reward, fake2




