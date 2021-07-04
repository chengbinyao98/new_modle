import matplotlib.pylab as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np

# # s_mu
# fig ,ax = plt.subplots(figsize = (7, 5))
#
# x11 = [-1,  0, 0.75, 1.5, 2.5]
# y11 = [98.96, 97.2, 93.5, 91.8, 89.6]
# x12 = [-1,  0, 0.75, 1.5, 2.5]
# y12 = [55.5, 56.3, 54.9, 57.8, 56.6]
# x13 = [-1,  0, 0.75, 1.5, 2.5]
# y13 = [20.3, 23.5, 26.4, 27.8, 29]
#
# x21 = [0.5,  1.5, 2, 3, 5]
# y21 = [99.8, 99.2, 98.5, 96.4, 93.2]
# x22 = [0.5,  1.5, 2, 3, 5]
# y22 = [97.5, 98.2, 95.2, 93.8, 92.1]
# x23 = [0.5,  1.5, 2, 3, 5]
# y23 = [84.6, 75.8, 75.3, 72.6, 70]
#
# x31 = [0.5,  1.5, 2.5, 4, 6]
# y31 = [98.9, 98.2, 97.27, 96.2, 95.1]
# x32 = [0.5,  1.5, 2.5, 4, 6]
# y32 = [98.5, 98.4, 97.2, 96.7, 95.2]
# x33 = [0.5,  1.5, 2.5, 4, 6]
# y33 = [91.11, 90.62, 88.5, 87.4, 85.7]
#
#
# plt.plot(x11,y11, label = "period1_multi", c='b', marker='+')
# plt.plot(x12,y12, label = "period1_single", c='r', marker='+')
# plt.plot(x13,y13, label = "period1_direct", c='g', marker='+')
#
# plt.plot(x21,y21, label = "period2_multi", c='b', marker='X')
# plt.plot(x22,y22, label = "period2_single", c='r', marker='X')
# plt.plot(x23,y23, label = "period2_direct", c='g', marker='X')
#
# plt.plot(x31,y31, label = "period3_multi", c='b', marker='d')
# plt.plot(x32,y32, label = "period3_single", c='r', marker='d')
# plt.plot(x33,y33, label = "period3_direct", c='g', marker='d')
#
# #
# # x = [1,2,3,4,5]
# # plt.plot(x,y11, label = "period1_multi", c='b', marker='+')
# # plt.plot(x,y12, label = "period1_single", c='r', marker='+')
# # plt.plot(x,y13, label = "period1_direct", c='g', marker='+')
# #
# # plt.plot(x,y21, label = "period2_multi", c='b', marker='X')
# # plt.plot(x,y22, label = "period2_single", c='r', marker='X')
# # plt.plot(x,y23, label = "period2_direct", c='g', marker='X')
# # #
# # plt.plot(x,y31, label = "period3_multi", c='b', marker='d')
# # plt.plot(x,y32, label = "period3_single", c='r', marker='d')
# # plt.plot(x,y33, label = "period3_direct", c='g', marker='d')
#
#
# plt.xlabel("s_mu", fontsize = 12)
# plt.ylabel("success rate", fontsize=12)
#
# # x_major_locator = MultipleLocator(1)
# # # 把x轴的刻度间隔设置为1，并存在变量里
# # y_major_locator = MultipleLocator(1)
# # # 把y轴的刻度间隔设置为10，并存在变量里
# # ax = plt.gca()
# # # ax为两条坐标轴的实例
# # ax.xaxis.set_major_locator(x_major_locator)
# # # 把x轴的主刻度设置为1的倍数
# # ax.yaxis.set_major_locator(y_major_locator)
# # # 把y轴的主刻度设置为10的倍数
#
#
# # plt.xticks(np.arange(7, 17, 1),fontproperties = 'Times New Roman', size = 10)
# # plt.yticks(np.arange(78, 100, 2),fontproperties = 'Times New Roman', size = 10)
#
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
#
# # plt.legend(bbox_to_anchor=(1.05, 1))
# lgnd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
# lgnd.legendHandles[0]._legmarker.set_markersize(16)
# lgnd.legendHandles[1]._legmarker.set_markersize(10)
#
# fig.subplots_adjust(right=0.7)
#
# plt.show()

#
# # s_sigma
# fig ,ax = plt.subplots(figsize = (7, 5))
#
# x11 = [0.05, 0.1, 0.25, 0.8]
# y11 = [92.6, 90.9, 88.6, 90.2]
# x12 = [0.05, 0.1, 0.25, 0.8]
# y12 = [55.4, 53.8, 56.6, 54.7]
# x13 = [0.05, 0.1, 0.25, 0.8]
# y13 = [30.2, 29.7, 29, 31.3]
#
# x21 = [0.02, 0.05, 0.1, 0.25]
# y21 = [98.3, 97.6, 96.8, 98.5]
# x22 = [0.02, 0.05, 0.1, 0.25]
# y22 = [93.3, 92.8, 93.5, 95.2]
# x23 = [0.02, 0.05, 0.1, 0.25]
# y23 = [76.2, 74.9, 73.3, 75.3]
#
# x31 = [0.01, 0.02, 0.05, 0.25]
# y31 = [98.56, 96.34, 94.76, 97.27]
# x32 = [0.01, 0.02, 0.05, 0.25]
# y32 = [98.2, 96.58, 94.4, 97.2]
# x33 = [0.01, 0.02, 0.05, 0.25]
# y33 = [90.6, 89.84, 87.98, 88.5]
#
#
# plt.plot(x11,y11, label = "period1_multi", c='b', marker='+')
# plt.plot(x12,y12, label = "period1_single", c='r', marker='+')
# plt.plot(x13,y13, label = "period1_direct", c='g', marker='+')
#
# plt.plot(x21,y21, label = "period2_multi", c='b', marker='X')
# plt.plot(x22,y22, label = "period2_single", c='r', marker='X')
# plt.plot(x23,y23, label = "period2_direct", c='g', marker='X')
# #
# plt.plot(x31,y31, label = "period3_multi", c='b', marker='d')
# plt.plot(x32,y32, label = "period3_single", c='r', marker='d')
# plt.plot(x33,y33, label = "period3_direct", c='g', marker='d')
# #
# # x = [1,2,3,4]
# # plt.plot(x,y11, label = "period1_multi", c='b', marker='+')
# # plt.plot(x,y12, label = "period1_single", c='r', marker='+')
# # plt.plot(x,y13, label = "period1_direct", c='g', marker='+')
# #
# # plt.plot(x,y21, label = "period2_multi", c='b', marker='X')
# # plt.plot(x,y22, label = "period2_single", c='r', marker='X')
# # plt.plot(x,y23, label = "period2_direct", c='g', marker='X')
# # #
# # plt.plot(x,y31, label = "period3_multi", c='b', marker='d')
# # plt.plot(x,y32, label = "period3_single", c='r', marker='d')
# # plt.plot(x,y33, label = "period3_direct", c='g', marker='d')
#
# plt.xlabel("s_sigma", fontsize = 12)
# plt.ylabel("success rate", fontsize=12)
#
# # x_major_locator = MultipleLocator(1)
# # # 把x轴的刻度间隔设置为1，并存在变量里
# # y_major_locator = MultipleLocator(1)
# # # 把y轴的刻度间隔设置为10，并存在变量里
# # ax = plt.gca()
# # # ax为两条坐标轴的实例
# # ax.xaxis.set_major_locator(x_major_locator)
# # # 把x轴的主刻度设置为1的倍数
# # ax.yaxis.set_major_locator(y_major_locator)
# # # 把y轴的主刻度设置为10的倍数
#
#
# # plt.xticks(np.arange(7, 17, 1),fontproperties = 'Times New Roman', size = 10)
# # plt.yticks(np.arange(78, 100, 2),fontproperties = 'Times New Roman', size = 10)
#
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
#
# # plt.legend(bbox_to_anchor=(1.05, 1))
# lgnd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
# lgnd.legendHandles[0]._legmarker.set_markersize(16)
# lgnd.legendHandles[1]._legmarker.set_markersize(10)
#
# fig.subplots_adjust(right=0.7)
#
# plt.show()







# d_sigma
fig ,ax = plt.subplots(figsize = (7, 5))

x11 = [0.05,0.1,0.25]
y11 = [90.34, 89.77, 88.6]
x12 = [0.05,0.1,0.25]
y12 = [55.4, 54.8, 56.6]
x13 = [0.05,0.1,0.25]
y13 = [28.3, 29.7,29]

x21 = [0.05,0.1,0.25]
y21 = [99.96, 99.2, 98.5]
x22 = [0.05,0.1,0.25]
y22 = [98.9, 98.3, 95.2]
x23 = [0.05,0.1,0.25]
y23 = [80.3, 78.7, 75.3]

x31 = [0.05,0.1,0.25]
y31 = [97.5, 97.38, 97.27]
x32 = [0.05,0.1,0.25]
y32 = [97.2, 97.4, 97.2]
x33 = [0.05,0.1,0.25]
y33 = [87.46, 88.4, 88.5]


plt.plot(x11,y11, label = "period1_multi", c='b', marker='+')
plt.plot(x12,y12, label = "period1_single", c='r', marker='+')
plt.plot(x13,y13, label = "period1_direct", c='g', marker='+')

plt.plot(x21,y21, label = "period2_multi", c='b', marker='X')
plt.plot(x22,y22, label = "period2_single", c='r', marker='X')
plt.plot(x23,y23, label = "period2_direct", c='g', marker='X')

plt.plot(x31,y31, label = "period3_multi", c='b', marker='d')
plt.plot(x32,y32, label = "period3_single", c='r', marker='d')
plt.plot(x33,y33, label = "period3_direct", c='g', marker='d')

# x = [1,2,3]
# plt.plot(x,y11, label = "period1_multi", c='b', marker='+')
# plt.plot(x,y12, label = "period1_single", c='r', marker='+')
# plt.plot(x,y13, label = "period1_direct", c='g', marker='+')
#
# plt.plot(x,y21, label = "period2_multi", c='b', marker='X')
# plt.plot(x,y22, label = "period2_single", c='r', marker='X')
# plt.plot(x,y23, label = "period2_direct", c='g', marker='X')
#
# plt.plot(x,y31, label = "period3_multi", c='b', marker='d')
# plt.plot(x,y32, label = "period3_single", c='r', marker='d')
# plt.plot(x,y33, label = "period3_direct", c='g', marker='d')

plt.xlabel("distance_sigma", fontsize = 12)
plt.ylabel("success rate", fontsize=12)

# x_major_locator = MultipleLocator(1)
# 把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator = MultipleLocator(1)
# # 把y轴的刻度间隔设置为10，并存在变量里
# ax = plt.gca()
# # ax为两条坐标轴的实例
# ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数/
# ax.yaxis.set_major_locator(y_major_locator)
# # 把y轴的主刻度设置为10的倍数


# plt.xticks(np.arange(7, 17, 1),fontproperties = 'Times New Roman', size = 10)
# plt.yticks(np.arange(78, 100, 2),fontproperties = 'Times New Roman', size = 10)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])

# plt.legend(bbox_to_anchor=(1.05, 1))
lgnd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, numpoints=1, fontsize=10)
lgnd.legendHandles[0]._legmarker.set_markersize(16)
lgnd.legendHandles[1]._legmarker.set_markersize(10)

fig.subplots_adjust(right=0.7)

plt.show()