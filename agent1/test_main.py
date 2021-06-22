from agent1.test_environment import Env1
from agent1.dqn1 import *
import matplotlib.pyplot as plt
import numpy as np
from draw import DRAW
import math

class Main1(object):
    def __init__(self,g):
        self.env = Env1()
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

    def train(self):
        plt.ion()
        plt.figure(figsize=(100, 5))  # 设置画布大小
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        # reward图
        epi = []
        success = []

        for episode in range(1000):
            step = 0

            print('episode', episode)
            epi.append(episode)

            state = self.env.reset()

            total_reward = 0
            total = 0

            while True:
                action = []
                add_action = []
                for i in range(len(state)):
                    add_action.append(self.rl.choose_action(np.array(state[i])))  # 学习到车组的动作组合
                    action.append(self.env.cars_posit[i] - self.env.road_range / 2 + add_action[i] * self.env.action_section)

                # self.draw.piant(self.env.cars_posit, self.env.road_range, ax1, self.env.frame_slot, action)

                state_, reward, learn_state_ = self.env.step(action, state)  # dicreward改成一个值
                if step == 10:
                    done = 1
                else:
                    done = 0
                step += 1


                for i in range(len(state)):
                    self.rl.store_transition_and_learn(state[i], add_action[i], reward[i], learn_state_[i], done)

                state = state_
                for i in range(len(reward)):
                    total_reward += reward[i]
                    total += self.env.beam_slot

                if done:
                    self.rl.saver_net()
                    break

            plt.sca(ax2)
            ax2.cla()
            # plt.ylim(0.6, 1.05)
            success.append(total_reward / total)
            plt.plot(epi, success)
            plt.pause(self.env.frame_slot )

    # def restore(self):
    #     self.rl.restore_net()

#
#
if __name__ == '__main__':
    g = tf.Graph()
    main = Main1(g)
    main.train()






