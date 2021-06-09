import matplotlib.pyplot as plt
import numpy as np
import math

if __name__ == '__main__':
    road_length = 200
    straight = 100
    ann_num = 32

    # plt.axis([0, 210, 0, 270])  # 坐标轴范围
    pos1 = 0
    pos2 = pos1 + 4

    road_range =50

    plt.figure(figsize=(40, 3))  # 设置画布大小

    plt.scatter(pos1, 0, marker="o")  # 画图数据
    plt.scatter(pos2, 0, marker="o")  # 画图数据



    act1 = []
    SNR1 = []
    for i in range(road_range):

        a = abs(road_length / 2 - pos1)
        # 斜边
        b = np.sqrt(np.square(a) + np.square(straight))
        if pos1 > road_length / 2:
            th1 = math.pi - math.acos(a / b)
        else:
            th1 = math.acos(a / b)
        channel = []
        for t in range(ann_num):
            m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
            channel.append(m.conjugate())

        act1.append(int(pos1) - road_range/2 + i)
        # 直角边
        c = abs(road_length / 2 - act1[i] )
        # 斜边
        d = np.sqrt(np.square(c) + np.square(straight))
        if act1[i] > road_length / 2:
            th2 = math.pi - math.acos(c / d)
        else:
            th2 = math.acos(c / d)
        signal = []
        for t in range(ann_num):
            n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
            signal.append(n)

        SNR1.append(np.square(np.linalg.norm(np.dot(channel,signal))))

    plt.plot(act1, SNR1)

    act2 = []
    SNR2 = []
    for i in range(road_range):

        a = abs(road_length / 2 - pos2)
        # 斜边
        b = np.sqrt(np.square(a) + np.square(straight))
        if pos2 > road_length / 2:
            th1 = math.pi - math.acos(a / b)
        else:
            th1 = math.acos(a / b)
        channel = []
        for t in range(ann_num):
            m = complex(math.cos(math.pi * t * math.cos(th1)), -math.sin(math.pi * t * math.cos(th1)))
            channel.append(m.conjugate())

        act2.append(int(pos2) - road_range/2 + i)
        # 直角边
        c = abs(road_length / 2 - act2[i] )
        # 斜边
        d = np.sqrt(np.square(c) + np.square(straight))
        if act2[i] > road_length / 2:
            th2 = math.pi - math.acos(c / d)
        else:
            th2 = math.acos(c / d)
        signal = []
        for t in range(ann_num):
            n = complex(math.cos(math.pi * t * math.cos(th2)), -math.sin(math.pi * t * math.cos(th2)))
            signal.append(n)

        SNR2.append(np.square(np.linalg.norm(np.dot(channel,signal))))

    plt.plot(act2, SNR2)

    plt.show()
