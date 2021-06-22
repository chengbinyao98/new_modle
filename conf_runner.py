import shutil
import os

prefx = "C:/Users/Administrator/PycharmProjects/new_modle/"

if __name__ == '__main__':

    for i in range(4):
        period = 1
        option = 4 - i
        import agent1.main1, agent2.main2, direct, main, one_agent
        print("-----------------------------------------")
        agent2.main2.run(period, option)
        agent1.main1.run(period, option)
        main.run(period, option)
        one_agent.run(period, option)
        direct.run(period, option)
        file = prefx + "6-22-Results/Period-{}___Condition-{}/".format(period, option)
        model_file = file + "model/"
        shutil.move("data", model_file)
        os.mkdir("data")
        print("-----------------------------------------")