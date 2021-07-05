import tensorflow as tf
from tool import Tools
from environment import Env
from draw import DRAW
import matplotlib.pyplot as plt
import numpy as np


def run():
    env = Env()
    tools = Tools()
    draw = DRAW()

    # g1 = tf.Graph()
    # main1 = Main1(g1)
    #
    # g2 = tf.Graph()
    # main2 = Main2(2, g2)

    plt.ion()
    plt.figure(figsize=(100, 5))  # 设置画布大小
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    success = 0
    totally = 0
    zongzhou = []


    while True:
        # main1.rl.restore_net()
        # main2.rl.restore_net()

        dic_state, fake = env.reset(tools)
        for episodes in range(1000):
            dic_action = {}
            suss = 0
            total = 0

            for x in dic_state:
                if x not in dic_action:
                    dic_action[x] = []

                if x == 1:
                    for num in range(len(dic_state[1])):
                        # temp_state = tools.get_list(dic_state[1][num])  # 车组中所有车辆状态合成
                        # temp = main1.rl.real_choose_action(temp_state)  # 学习到车组的动作组合
                        dic_action[1].append([dic_state[1][num][0][0]])

                if x == 2:
                    for num in range(len(dic_state[2])):
                        action = []
                        for dim in range(2):
                            action.append(dic_state[2][num][dim][0])
                        dic_action[2].append(action)

            # draw_action = [0 for l in range(len(env.cars_posit))]
            # for x in dic_state:
            #     for num in range(len(dic_state[x])):
            #         for dim in range(len(dic_state[x][num])):
            #             draw_action[dic_state[x][num][dim][3]] = dic_action[x][num][dim]
            # draw.piant(env.cars_posit,env.road_range,ax1,env.frame_slot,draw_action)

            dic_state_, dic_reward, fake2 = env.step(dic_action, fake, tools)
            print(dic_reward)

            for x in dic_reward:
                for num in range(len(dic_reward[x])):
                    for dim in range(x):
                        suss += dic_reward[x][num][dim]
                        total += env.beam_slot
            print('成功率',suss/total)

            dic_state = dic_state_

            success += suss
            totally += total
            zongzhou.append(success/totally)

            plt.sca(ax2)
            ax2.cla()
            plt.plot([i for i in range(len(zongzhou))], zongzhou)
            plt.pause(env.frame_slot)

        from cluster_runner import prefx, fname
        file = prefx + fname + "/image/"
        import os
        if not os.path.exists(file):
            os.mkdir(file)
        plt.savefig(file + "direct.png")
        with open(file + "success_rate.txt", "a") as f:
            f.write("direct: {}\n".format(suss / total))
        plt.close()
        break