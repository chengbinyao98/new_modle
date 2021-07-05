# 何竞择
import pandas as pd



# 选择三种配置中的哪一种，以及其中的第几组数据。
from s_mu import Mean
m = Mean(1)
# condition = pd.read_csv("image_results/new_results.csv")

if __name__ == '__main__':
    import agent1.main1, agent2.main2, direct, main, one_agent
    print("-----------------------------------------")
    # agent2.main2.run()
    agent1.main1.run()
    # main.run()
    # one_agent.run()
    # direct.run()
    print("-----------------------------------------")