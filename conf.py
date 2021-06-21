# 何竞择
import pandas as pd

datasource = pd.read_csv("image_results/datasource.csv")

s_point1_num = 0
s_point2_num = 20
s_point1 = s_point1_num * 0.2777778
s_point2 = s_point2_num * 0.2777778
d_point1 = 4
d_point2 = 10
s_mu = 2.5
s_sigma = 0.25
d_sigma = 2

# if __name__ == '__main__':
#     import agent1.main1, agent2.main2, direct, main, one_agent
#     print("-----------------------------------------")
#     agent2.main2.run()
#     agent1.main1.run()
#     main.run()
#     one_agent.run()
#     direct.run()
#     print("-----------------------------------------")