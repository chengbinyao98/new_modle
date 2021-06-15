# 何竞择

s_point1 = 40 * 0.2777778
s_point2 = 60 * 0.2777778
d_point1 = 30
d_point2 = 60
s_mu = 1.5
s_sigma = 0.25
d_sigma = 10

if __name__ == '__main__':
    import agent1.main1, agent2.main2, direct, main, one_agent
    print("-----------------------------------------")
    agent2.main2.run()
    agent1.main1.run()
    main.run()
    one_agent.run()
    direct.run()
    print("-----------------------------------------")