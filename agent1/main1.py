from agent1.environment1 import Env1
from agent1.dqn1 import *
import matplotlib.pyplot as plt
import numpy as np
from agent1.draw1 import DRAW
import math

class Main1(object):
    def __init__(self,g, period, option):
        self.period = period
        self.option = option
        self.env = Env1(option)
        self.draw = DRAW()
        self.rl = DQN1(
                gra=g,
                s_dim=2,
                a_dim=int(math.ceil(self.env.road_range/self.env.action_section)),
                batch_size=128,
                gamma=0.99,
                lr=0.01,
                epsilon=0.1,
                replace_target_iter=300
                )

    def train(self, factor):
        plt.ion()
        plt.figure(figsize=(100, 5))  # 设置画布大小
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        # reward图
        epi = []
        success = []
        sf = False
        su_avg = 0
        for episode in range(1000):
            print('episode', episode)
            epi.append(episode)

            state = self.env.reset()

            total_reward = 0
            time = 0

            while True:
                add_action = self.rl.choose_action(np.array(state))  # 学习到车组的动作组合
                action = self.env.cars_posit - self.env.road_range / 2 + add_action * self.env.action_section

                # self.draw.piant(self.env.cars_posit, self.env.road_range, ax1, self.env.frame_slot, action)

                state_, reward, done = self.env.step(action, state)  # dicreward改成一个值

                self.rl.store_transition_and_learn(state, add_action, reward, state_, done)

                state = state_
                total_reward += reward
                time += 1
                if done:
                    self.rl.saver_net()
                    break

            plt.sca(ax2)
            ax2.cla()
            plt.ylim(0.6, 1.05)
            success.append(total_reward / (self.env.beam_slot * time))
            plt.plot(epi, success)
            plt.pause(self.env.frame_slot)

            if episode >= 30:
                if not sf:
                    su_avg = np.mean(success)
                    if su_avg > factor:
                        sf = True
                    else:
                        return False
        from conf_runner import prefx
        file = prefx + "6-22-Results/Period-{}___Condition-{}/".format(self.period, self.option)
        plt.savefig(file + "image/main1.png")
        with open(file + "image/success_rate.txt", "a") as f:
            f.write("main1: {}\n".format(su_avg))
        return True

    # def restore(self):
    #     self.rl.restore_net()

#
#
def run(period, option):
    flag = False
    count = 0
    factor = 0.9
    while not flag:
        g = tf.Graph()
        main = Main1(g, period, option)
        flag = main.train(factor)
        count += 1
        if count >= 9:
            factor = factor - 0.1
            count = 0






