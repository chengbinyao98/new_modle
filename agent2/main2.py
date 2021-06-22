from agent2.environment2 import Env2
from agent2.tools2 import Tools
from agent2.dqn2 import *
import matplotlib.pyplot as plt
import numpy as np
import math
from agent2.draw2 import DRAW

class Main2(object):
    def __init__(self, n, g, option):
        self.n = n
        self.env = Env2(option)
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

    def train(self):
        # 画图
        plt.ion()
        plt.figure(figsize=(100, 5))    # 设置画布大小
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        # reward图
        epi = []
        success = []


        for episode in range(20):
            sf = False
            print('episode',episode)
            epi.append(episode)

            total_reward = 0
            time = 0

            state = self.env.reset(self.n)

            while True:

                temp_state = self.tools.get_list(state)  # 车组中所有车辆状态合成
                add_action = self.rl.choose_action(np.array(temp_state))    # 学习到车组的动作组合
                add_action1 = self.rl.choose_action(np.array(temp_state))    # 学习到车组的动作组合

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
                for dim in range(self.n):
                    action.append(self.env.cars_posit[dim] - self.env.road_range / 2 + add[dim] * self.env.action_section)

                # self.draw.piant(self.env.cars_posit, self.env.road_range, ax1, self.env.frame_slot, self.n, action)

                state_, reward, done = self.env.step(action,state,self.n)  # dicreward改成一个值

                l_temp_state = self.tools.get_list(state)
                l_temp_state_ = self.tools.get_list(state_)
                self.rl.store_transition_and_learn(l_temp_state, add_action, reward, l_temp_state_, done)

                total_reward += reward
                time += 1

                state = state_
                if done:
                    self.rl.saver_net()
                    break

            plt.sca(ax2)
            ax2.cla()
            success.append(total_reward/(self.env.beam_slot*time*self.n))
            plt.plot(epi, success)
            plt.pause(self.env.frame_slot)

            if episode >= 10:
                if not sf:
                    su_avg = np.mean(success)
                    if su_avg > 0.4:
                        sf = True
                    else:
                        return False
        from conf_runner import file
        plt.savefig(file + "image/main2.png")
        with open(file + "image/success_rate.txt", "a") as f:
            f.write("main2: {}\n".format(su_avg))
        return True


def run(option):
    flag = False
    while not flag:
        g = tf.Graph()
        main = Main2(2,g, option)
        flag = main.train()


