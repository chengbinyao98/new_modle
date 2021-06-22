import shutil

period = 1
option = 5
prefx = "C:/Users/jingz/Desktop/new_modle/"
file = prefx + "6-22-Results/Period-{}___Condition-{}/".format(period, option)
model_file = file + "model/"

if __name__ == '__main__':


    import agent1.main1, agent2.main2, direct, main, one_agent
    print("-----------------------------------------")
    agent2.main2.run(option)
    agent1.main1.run(option)
    main.run(option)
    one_agent.run(option)
    direct.run(option)
    shutil.move("data", model_file)
    print("-----------------------------------------")