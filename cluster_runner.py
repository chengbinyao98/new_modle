# 何竞择
import os
import shutil



for i in range(1,3):
    # 参数选择！！！！！！！！！！
    if not os.path.exists("data"):
        os.mkdir("data")

    if not os.path.exists("Result"):
        os.mkdir("Result")

    op = i + 1
    conf_name = "d_sigma"
    if conf_name == "s_mu":
        from s_mu import Mean
    if conf_name == "s_sigma":
        from s_sigma import Mean
    if conf_name == "d_sigma":
        from d_sigma import Mean

    # 指定路径
    prefx = "C:/Users/Administrator/PycharmProjects/new_modle/"
    m = Mean(op)
    # 选择三种配置中的哪一种，以及其中的第几组数据。

    fname = conf_name + ",第" + str(op) + "组"

    import agent1.main1, agent2.main2, direct, main, one_agent

    print("-----------------------------------------")
    if op == 2:
        direct.run()
    else:
        agent2.main2.run()
        agent1.main1.run()
        main.run()
        one_agent.run()
        direct.run()

    # 处理模型文件
    model_file = prefx + fname + "/model/"
    if not os.path.exists(prefx + fname):
        os.mkdir(prefx + fname)
    if not os.path.exists(model_file):
        os.mkdir(model_file)

    shutil.move("data", model_file)
    os.mkdir("data")
    print("-----------------------------------------")
