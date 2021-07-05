from agent2.environment2 import Env2
from agent2.tools2 import Tools
from agent2.dqn2 import *
import matplotlib.pyplot as plt
import numpy as np
import math
from agent2.draw2 import DRAW

class Main2(object):
    def __init__(self, n, g):
        self.n = n
        self.env = Env2()
        self.tools = Tools()
        self.draw = DRAW()
        self.rl = DQN2(
            gra=g,
            s_dim=2 * self.n,
            a_dim=int(math.pow(int(math.ceil(self.env.road_range/self.env.action_section)),n)),
            batch_size=128,
            gamma=0.99,
            lr=0.01,
            epsilon=0.1,
            replace_target_iter=300
        )

    def train(self, factor):
        # 画图
        plt.ion()
        plt.figure(figsize=(100, 5))    # 设置画布大小
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        # reward图
        epi = []
        success = []
        sf = False


        for episode in range(1000):
            print('episode',episode)
            epi.append(episode)

            total_reward = 0
            time = 0

            state,fake = self.env.reset(self.n)

            while True:

                add_action = self.rl.choose_action(np.array(state))    # 学习到车组的动作组合
                add_action1 = self.rl.choose_action(np.array(state))    # 学习到车组的动作组合

                # 车组动作组合转换成车辆的单个动作增量
                add = []
                b = []
                for k in range(self.n):
                    s = add_action1 // math.ceil(self.env.road_range/self.env.action_section)  # 商
                    y = add_action1 % math.ceil(self.env.road_range/self.env.action_section)  # 余数
                    b = b + [y]
                    add_action1 = s
                b.reverse()
                for i in b:
                    add.append(i)

                # 转换成车辆的单个动作
                action = []
                for dim in range(2):
                    action.append(state[dim] - self.env.road_range / 2 + add[dim] * self.env.action_section)

                # self.draw.piant(fake[0], self.env.road_range, ax1, self.env.frame_slot, self.n, action)

                state_, reward, done, fake2 = self.env.step(action,fake,self.n)  # dicreward改成一个值

                self.rl.store_transition_and_learn(state, add_action, reward, state_, done)

                total_reward += reward
                time += 1

                state = state_
                fake = fake2
                if done:
                    self.rl.saver_net()
                    break
            plt.sca(ax2)
            ax2.cla()
            success.append(total_reward/(self.env.beam_slot*time*self.n))
            plt.plot(epi, success)
            plt.pause(self.env.frame_slot)

            if episode >= 50:
                if not sf:
                    su_avg = np.mean(success)
                    if su_avg > factor:
                        sf = True
                    else:
                        return False
        from cluster_runner import prefx, fname
        file = prefx + fname + "/image/"
        import os
        if not os.path.exists(prefx + fname):
            os.mkdir(prefx + fname)
        if not os.path.exists(file):
            os.mkdir(file)
        plt.savefig(file + "main2.png")
        with open(file + "success_rate.txt", "a") as f:
            f.write("main2: {}\n".format(su_avg))
        plt.close()
        return True


def run():
    flag = False
    count = 0
    factor = 0.65
    while not flag:
        g = tf.Graph()
        main = Main2(2,g)
        flag = main.train(factor)
        count += 1
        if count >= 9:
            factor = factor - 0.1
            count = 0




