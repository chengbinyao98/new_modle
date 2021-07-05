# import numpy as np
# import math
# from pylab import *
# from scipy import stats
# from mean import Mean
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.mlab as mlab
# from scipy.stats import norm
#
#
# class Env1(object):
#     def __init__(self):
#         m = Mean()
#         # 固定量
#         # 帧结构
#         self.frame_slot = 0.01          # 帧时隙时间长度
#         self.beam_slot = 100            # 波束选择时隙数
#         self.right = 5                  # 正确传输最低的SNR
#         # 车辆和道路
#         self.road_length = 200          # 道路长度
#         self.straight = 100             # 基站和道路的直线距离
#
#         # 存储单元
#         self.cars_posit = 0  # 车辆的位置
#         self.cars_speed = 0  # 车辆的速度
#
#         # 变化量
#         # 算法变化量
#         self.road_section = 2            # 每几米划分成一个路段
#         self.action_section = 1          # 每几米划分成一个路段
#         self.road_range = 35             # 动作可以选择的范围
#
#         # 通信变化量
#         self.ann_num = 32                # 天线数目
#
#         # 道路变化量
#         self.s_mu, self.s_sigma, self.d_mu, self.d_sigma, self.s_point1, self.s_point2, self.d_point1, self.d_point2 = m.time1()
#
#         # # 车速分布
#         # self.d_sigma = 2                                                        # 车辆间距分布
#         #
#         # # 同一个时段不用变化
#         # self.s_point1 = 0 * 0.277777778                                         # 车速范围
#         # self.s_point2 = 20 * 0.277777778
#         # self.d_point1 = 4                                                       # 车间距范围
#         # self.d_point2 = 10
#         #
#         # safe_dis = self.d_point1                                                # 安全距离
#         # km = self.road_length / (self.car_length + safe_dis)                    # 由安全距离计算最大车辆密度
#         #
#         # data = self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)
#         # mean = np.mean(data)
#         # k = km / math.exp(mean / self.max_speed)
#         # distance = self.road_length / k - self.car_length
#         # self.d_mu = math.log(distance) - self.d_sigma * self.d_sigma / 2
#
#     # 由道路上的所有车辆得到所有车辆的路段
#     def get_section(self, pos):
#         section = math.ceil(pos / self.road_section)
#         return section
#
#     def log_zhengtai(self, mu, sigma, log_lower, log_upper, data_num=1):
#         norm_lower = np.log(log_lower)
#         norm_upper = np.log(log_upper)
#         X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
#         norm_data = X.rvs(data_num)
#         log_data = np.exp(norm_data)
#         return log_data
#
#
#     # def exp(self, scale, low, high, data_num=1):  # scale是均值不是lamda，是1/lamda
#     #     rnd_cdf = np.random.uniform(stats.expon.cdf(x=low, scale=scale),
#     #                                 stats.expon.cdf(x=high, scale=scale),
#     #                                 size=data_num)
#     #     return stats.expon.ppf(q=rnd_cdf, scale=scale)
#
#     def get_reward(self, act, reward):
#         # 直角边
#         a = abs(self.road_length / 2 - self.cars_posit)
#         # 斜边
#         b = np.sqrt(np.square(a) + np.square(self.straight))
#         if self.cars_posit > self.road_length / 2:
#             th1 = math.pi - math.acos(a / b)
#         else:
#             th1 = math.acos(a / b)
#
#         channel = []
#         for t in range(self.ann_num):
#             m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
#             channel.append(m.conjugate())
#
#         # 直角边
#         c = abs(self.road_length / 2 - act)
#         # 斜边
#         d = np.sqrt(np.square(c) + np.square(self.straight))
#         if act > self.road_length / 2:
#             th2 = math.pi - math.acos(c / d)
#         else:
#             th2 = math.acos(c / d)
#
#         signal = []
#         for t in range(self.ann_num):
#             n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
#             signal.append(n)
#
#         SNR = np.square(np.linalg.norm(np.dot(channel, signal)))
#
#         if SNR >= self.right:
#             reward += 1
#         return reward
#
#     def reset(self):
#         # 道路环境初始化
#         self.cars_speed = self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)[0]
#         self.cars_posit = self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]
#         # 形成状态
#         a = self.get_section(self.cars_posit)
#         state = [a,a]
#         return state
#
#     def step(self, action, state):
#
#         # 道路的（位置更新）
#         reward = 0
#         for i in range(self.beam_slot):  # 标记当前车辆是否之前被操作过，保证一个时隙车只跑一个时隙的量
#             self.cars_posit += self.cars_speed * self.frame_slot
#             reward = self.get_reward(action,reward)
#
#         state_ =[self.get_section(self.cars_posit), state[0]]
#
#         if self.cars_posit > self.road_length:
#             done = 1
#         else:
#             done = 0
#
#         return state_, reward, done

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from pylab import *
from scipy import stats
from cluster_runner import m

import matplotlib
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
        self.s_mu, self.s_sigma, self.d_mu, self.d_sigma, self.s_point1, self.s_point2, self.d_point1, self.d_point2, \
        self.safe_dis, self.near_dis, self.far_dis, self.target_dis,\
        self.mean_speed,\
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
        count = 1/2
        for i in range(len(self.cars_posit)):
            if i != len(self.cars_posit) - 1:
                if self.cars_posit[i+1] - self.cars_posit[i] <= self.near_dis:
                    self.cars_posit[i] = self.cars_posit[i+1] - self.safe_dis - (self.cars_posit[i+1]-self.safe_dis-self.cars_posit[i]) / 2
                    self.cars_speed[i] = self.cars_speed[i+1]
                if self.cars_posit[i+1] - self.cars_posit[i] >= self.far_dis:
                    self.cars_posit[i] = self.cars_posit[i+1] - self.target_dis - (self.cars_posit[i+1] - self.target_dis - self.cars_posit[i]) / 2
                    self.cars_speed[i] = self.mean_speed + np.random.uniform(-0.5, 0.5)
                if self.near_dis < self.cars_posit[i + 1] - self.cars_posit[i] < self.far_dis :
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

    def get_reward(self, act, pos, reward):

    # 直角边
        a = abs(self.road_length / 2 - pos)
        # 斜边
        b = np.sqrt(np.square(a) + np.square(self.straight))
        if pos > self.road_length / 2:
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

    # def reset(self, tool):
    #     # 道路环境初始化
    #     self.road_reset()
    #     # 获得道路上的每个车辆信息
    #     info = tool.get_info(self.cars_posit, self.no_interference)
    #     number = [i for i in range(len(self.cars_posit))]
    #
    #     # 形成状态
    #     a = tool.classify(number, info)
    #     b = tool.classify(self.get_section(self.cars_posit), info)
    #     d = tool.integrate(b, b, a)
    #     return d

    def reset(self):
        # 道路环境初始化
        self.cars_speed = [self.log_zhengtai(self.s_mu, self.s_sigma, self.s_point1, self.s_point2)[0]]
        self.cars_posit = [self.log_zhengtai(self.d_mu, self.d_sigma, self.d_point1, self.d_point2)[0]]
        a = self.cars_posit[0]

        fake = []
        for i in range(self.beam_slot):
            self.road_step()
            fake.append(self.cars_posit[0])
        # 形成状态
        return [a,self.cars_posit[0]],fake

    def step(self, action, fake):

        # 道路的（位置更新）
        reward = 0
        fake2 = []
        for i in range(self.beam_slot):  # 标记当前车辆是否之前被操作过，保证一个时隙车只跑一个时隙的量
            reward = self.get_reward(action, fake[i], reward)
            self.road_step()
            fake2.append(self.cars_posit[0])

        state_ =[fake[self.beam_slot-1], self.cars_posit[0]]

        if fake[self.beam_slot-1] > self.road_length:
            done = 1
        else:
            done = 0

        return state_, reward, done, fake2

# if __name__ == '__main__':
#     env = Env1()
#     env.road_reset()
#     plt.ion()
#     # env.batch.append(env.cars_posit)
#     print('safe_dis', env.safe_dis,
#           'near',env.near_dis,
#           'far', env.far_dis,
#           'target', env.target_dis,
#          'speed', env.mean_speed,
#           'k',env.car_num)
#     plt.figure(figsize = (100, 5))  # 设置画布大小
#     for j in range(10000):
#
#         # a_ts = []
#         # b_ts = []
#         # for m in range(env.no_num_change):
#         #     a = []
#         #     for i in range(len(env.cars_posit)):
#         #         a.append(env.cars_posit[i])
#         #     a_ts.append(a)
#         #     # print(env.batch)
#         #
#         #     for k in range(env.beam_slot):
#         #         env.road_step()
#         #
#         #     b = []
#         #     for i in range(len(env.cars_posit)):
#         #         b.append(env.cars_posit[i])
#         #     b_ts.append(b)
#
#         plt.cla()
#         plt.xlim(0, 250)
#         y1 = [0 for i in range(len(env.cars_posit))]
#         # y2 = [1 for i in range(len(env.cars_posit))]
#         plt.scatter(env.cars_posit, y1, marker = "o")  # 画图数据
#         for i in range(len(y1)):
#             plt.annotate(format(env.cars_speed[i], '.2f'), xy = (env.cars_posit[i], y1[i]),
#                          xytext = (env.cars_posit[i] + 0.1, y1[i] + 0.01))
#         for i in range(len(y1) - 1):
#             plt.annotate(format((env.cars_posit[i + 1] - env.cars_posit[i]), '.2f'),
#                          xy = (env.cars_posit[i], y1[i]),
#                          xytext = ((env.cars_posit[i] + env.cars_posit[i + 1]) / 2, y1[i]))
#
#         # print(env.cars_posit)
#
#         plt.pause(1.5)
#
#
#         for i in range(env.beam_slot):
#             env.road_step()
#
#         plt.cla()
#         plt.xlim(0, 250)
#         y1 = [0 for i in range(len(env.cars_posit))]
#         # y2 = [1 for i in range(len(env.cars_posit))]
#         plt.scatter(env.cars_posit, y1, marker = "o")  # 画图数据
#         for i in range(len(y1)):
#             plt.annotate(format(env.cars_speed[i], '.2f'), xy = (env.cars_posit[i], y1[i]),
#                          xytext = (env.cars_posit[i] + 0.1, y1[i] + 0.01))
#         for i in range(len(y1) - 1):
#             plt.annotate(format((env.cars_posit[i + 1] - env.cars_posit[i]), '.2f'),
#                          xy = (env.cars_posit[i], y1[i]),
#                          xytext = ((env.cars_posit[i] + env.cars_posit[i + 1]) / 2, y1[i]))
#
#         # print(env.cars_posit)
#
#         plt.pause(1.5)
#
#         for i in range(env.beam_slot):
#             env.road_step()
#
#         plt.cla()
#         plt.xlim(0, 250)
#         y1 = [0 for i in range(len(env.cars_posit))]
#         # y2 = [1 for i in range(len(env.cars_posit))]
#         plt.scatter(env.cars_posit, y1, marker = "o")  # 画图数据
#         for i in range(len(y1)):
#             plt.annotate(format(env.cars_speed[i], '.2f'), xy = (env.cars_posit[i], y1[i]),
#                          xytext = (env.cars_posit[i] + 0.1, y1[i] + 0.01))
#         for i in range(len(y1) - 1):
#             plt.annotate(format((env.cars_posit[i + 1] - env.cars_posit[i]), '.2f'),
#                          xy = (env.cars_posit[i], y1[i]),
#                          xytext = ((env.cars_posit[i] + env.cars_posit[i + 1]) / 2, y1[i]))
#
#         # print(env.cars_posit)
#
#         plt.pause(1.5)
#
#         # print('a',a_ts)
#         # print('b',b_ts)
#         #
#         # if len(env.seq_batch) == env.batch_size:
#         #     env.seq_batch.append(a_ts)
#         #     env.res_batch.append(b_ts)
#         #     del env.seq_batch[0]
#         #     del env.res_batch[0]
#         # else:
#         #     env.seq_batch.append(a_ts)
#         #     env.res_batch.append(b_ts)
#         #
#         #
#         # print(len(env.cars_posit))
#
#
#
#         # if env.cars_posit[len(env.cars_posit) - 1] > env.road_length:
#         #     env.road_reset()
#         env.get_information()
#
#         #
#         #
#         #
#         # print('seq',env.seq_batch)
#         # print('res',env.res_batch)







