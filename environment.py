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
        # 固定量
        # 帧结构
        self.frame_slot = 0.01          # 帧时隙时间长度
        self.beam_slot = 100            # 波束选择时隙数
        self.right = 5                  # 正确传输最低的SNR
        # 车辆和道路
        self.road_length = 200          # 道路长度
        self.straight = 100             # 基站和道路的直线距离
        self.car_length = 5
        self.max_speed = 105

        # 存储单元
        self.cars_posit = []            # 车辆的位置（连续）
        self.cars_speed = []            # 车辆的速度（连续

        # 变化量
        # 算法变化量
        self.road_section = 2           # 每几米划分成一个路段
        self.action_section = 1         # 每几米划分成一个路段
        self.road_range = 35            # 动作可以选择的范围

        # 通信变化量
        self.ann_num = 32               # 天线数目
        self.no_interference = 30       # 随着天线个数变化

        # 道路变化量
        self.s_mu, self.s_sigma = 2.5, 0.25                                 # 车速分布
        self.d_sigma = 0.05                                                 # 车辆间距分布

        # 同一个时段不用变化
        self.s_point1 = 0                                                   # 车速范围
        self.s_point2 = 20*0.277777778
        self.d_point1 = 4                                                   # 车间距范围
        self.d_point2 = 10
        safe_dis = 4  # 安全距离
        km = self.road_length / (self.car_length + safe_dis)                # 由安全距离计算最大车辆密度
        mean = math.exp(self.s_mu + self.s_sigma * self.s_sigma / 2)        # 由交通流理论计算车辆间距
        k = km / math.exp(mean / self.max_speed)
        distance = self.road_length / k - self.car_length
        self.d_mu = math.log(distance) - self.d_sigma * self.d_sigma / 2

    # 由道路上的所有车辆得到所有车辆的路段
    def get_section(self, list):
        section = []
        for i in range(len(list)):
            section.append(math.ceil(list[i] / self.road_section))
        return section

    def log_zhengtai(self, mu, sigma, log_lower, log_upper, data_num = 1):
        norm_lower = np.log(log_lower)
        norm_upper = np.log(log_upper)
        X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc = mu, scale = sigma)
        norm_data = X.rvs(data_num)
        log_data = np.exp(norm_data)
        return log_data

    def exp(self, scale, low, high, data_num = 1):  # scale是均值不是lamda，是1/lamda
        rnd_cdf = np.random.uniform(stats.expon.cdf(x = low, scale = scale),
                                    stats.expon.cdf(x = high, scale = scale),
                                    size = data_num)
        return stats.expon.ppf(q = rnd_cdf, scale = scale)

    # 道路初始化
    def road_reset(self):
        speed = self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)[0]
        for i in range(50):  # 任意数目都可以，主要是用于生成路段上的车辆
            dis = self.log_zhengtai(self.d_mu, self.d_sigma, self.s_point1, self.s_point2)[0]
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

    # 道路路面现有车辆更新
    def road_step(self):
        for i in range(len(self.cars_posit)):
            self.cars_posit[i] = self.cars_speed[i] * self.frame_slot + self.cars_posit[i]


    def get_information(self, action, section):
        for i in range(10):  # 这个10随便，只要保证能新加上所有的车辆即可
            # 生成一个新的车辆进入，初始化车辆间距
            dis1 = self.log_zhengtai(self.d_mu, self.d_sigma, self.s_point1, self.s_point2)[0]
            dis2 = self.log_zhengtai(self.d_mu, self.d_sigma, self.s_point1, self.s_point2)[0]
            if self.cars_posit[0] >= dis1 + dis2:
                action.insert(0, (self.cars_posit[0] - dis1) / self.road_section)
                section.insert(0, (self.cars_posit[0] - dis1) / self.road_section)
                self.cars_posit.insert(0, (self.cars_posit[0] - dis1))  # 车辆的位置（位置更新）
                self.cars_speed.insert(0, self.cars_speed[0])  # 车辆的速度（位置更新）
            else:
                break
        for i in range(10):
            # 将超出道路的车辆排除
            if self.cars_posit[len(self.cars_posit) - 1] > self.road_length:
                del action[len(self.cars_posit) - 1]
                del section[len(self.cars_posit) - 1]
                del self.cars_speed[len(self.cars_posit) - 1]
                del self.cars_posit[len(self.cars_posit) - 1]
            else:
                break
        return action, section

    def get_reward(self, list_act, list_reward, tool):
        info = tool.get_info(self.cars_posit, self.no_interference)
        for i in range(len(list_act)):
            SNR_noise = 0
            SNR = 0
            for num in range(i - info[i][2], i - info[i][2] + info[i][0]):
                # 直角边
                a = abs(self.road_length / 2 - self.cars_posit[num])
                # 斜边
                b = np.sqrt(np.square(a) + np.square(self.straight))
                if self.cars_posit[num] > self.road_length / 2:
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
        # 道路环境初始化
        self.road_reset()
        # 获得道路上的每个车辆信息
        info = tool.get_info(self.cars_posit, self.no_interference)
        number = [i for i in range(len(self.cars_posit))]

        # 形成状态
        a = tool.classify(number, info)
        b = tool.classify(self.get_section(self.cars_posit), info)
        d = tool.integrate(b, b, a)
        return d

    def step(self, dic_action, tool):
        info = tool.get_info(self.cars_posit, self.no_interference)  # 当前道路的信息
        # print(info)
        # print(tool.get_info1(self.cars_posit, self.no_interference))
        action = tool.reverse_classify(dic_action, info)  # 当前车辆的动作
        section = self.get_section(self.cars_posit)  # 当前车辆的路段  用于产生下一时刻的状态

        # 道路的（位置更新）
        reward = [0 for p in range(len(info))]  # 用于记录一个帧周期的车辆情况
        for i in range(self.beam_slot):
            self.road_step()
            reward = self.get_reward(action, reward, tool)
        dic_reward = tool.classify(reward, info)

        action, change_next_section = self.get_information(action, section)

        next_info = tool.get_info(self.cars_posit, self.no_interference)  # 下一时刻的道路信息

        # 下一时刻的状态（位置更新）（数目更新）
        b = tool.classify(self.get_section(self.cars_posit), next_info)
        c = tool.classify(change_next_section, next_info)
        d = tool.classify([i for i in range(len(self.cars_posit))],next_info)
        dic_state_ = tool.integrate(b, c, d)

        return dic_state_, dic_reward

    def draw(self):

        plt.ion()  # 开启交互模式
        plt.figure(figsize=(100, 3))  # 设置画布大小

        # 数据
        y = []
        for i in range(len(self.cars_posit)):
            y.append(0)

        for j in range(1000):
            plt.clf()  # 清空画布
            plt.axis([0, 210, 0, 0.1])  # 坐标轴范围
            x_major_locator = MultipleLocator(5)  # 把x轴的刻度间隔设置为1，并存在变量里
            ax = plt.gca()  # ax为两条坐标轴的实例
            ax.xaxis.set_major_locator(x_major_locator)  # 把x轴的主刻度设置为1的倍数
            plt.tick_params(axis='both', which='major', labelsize=5)  # 坐标轴字体大小

            self.road_step()

            plt.scatter(self.cars_posit, y, marker="o")  # 画图数据
            plt.pause(0.2)

        plt.ioff()
        plt.show()


