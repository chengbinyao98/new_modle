import matplotlib.pylab as plt
import pandas as pd

def write_data(z, t):
    # 读取csv结果文件
    results = pd.read_csv("nine_periods.csv")
    print(results)
    from s_mu import Mean
    for option in range(z):
        # 读取配置参数
        m = Mean(option + 1)
        if t == 1:
            s_mu, s_sigma, d_mu, d_sigma, s_point1, s_point2, d_point1, d_point2 = m.time1()
        elif t == 2:
            s_mu, s_sigma, d_mu, d_sigma, s_point1, s_point2, d_point1, d_point2 = m.time2()
        elif t == 3:
            s_mu, s_sigma, d_mu, d_sigma, s_point1, s_point2, d_point1, d_point2 = m.time3()
        dict = {'s_mu': s_mu, 's_sigma': s_sigma, 'd_mu': d_mu, 'd_sigma': d_sigma}

        # 读取main2、main1成功率
        path = "../6-22-Results/Period-1___Condition-{}/image/success_rate.txt".format(option + 1)
        with open(path, 'r') as f:
            main2 = float(f.readline()[7:12])
            main1 = float(f.readline()[7:12])
            dict['main2'] = main2
            dict['main1'] = main1
        results = results.append(dict, ignore_index = True)
    results.to_csv("nine_periods.csv", index = False)


def print_figure(period_start_index, x_range, x_name, title):
    # 初始化
    x_axis = []
    x = []
    x_text = []
    main1 = []
    main2 = []
    y_main = []
    y_one_agent = []
    y_direct = []

    # 取时段内参数组数
    for i in range(x_range):
        x_axis.append(i)
        index = period_start_index + i
        x.append(results[x_name][index])
        main1.append(results['main1'][index])
        main2.append(results['main2'][index])
        y_main.append(results['main'][index])
        y_one_agent.append(results['one_agent'][index])
        y_direct.append(results['direct'][index])
        x_text.append("{}: {}\nmain1: {}\nmain2: {}".format(x_name,x[i],round(main1[i],3),round(main2[i],3)))

    # 绘图
    plt.plot(x_axis, y_main, 'xr-')
    plt.plot(x_axis, y_one_agent, 'oy-')
    plt.plot(x_axis, y_direct, '^b-')
    # x刻度归一化处理，并加上main1，main2的数据
    plt.xticks(x_axis, x_text)
    plt.ylabel("Success rate", fontsize = 12)
    plt.legend(['main', 'one_agent', 'direct'])
    # 添加标题
    plt.title(title)
    save_path = "C:/Users/jingz/Desktop/results_figure/{}-period_{}.png".format(x_name, t)
    plt.savefig(save_path)
    plt.show()



z = 5
t = 2


write_data(z, t)

# name = "s_mu"
# if t == 1:
#     title = "Period 1, v:[0-20], d:[4-10]"
# elif t == 2:
#     title = "Period 2, v:[20-40], d:[10-30]"
# elif t == 3:
#     title = "Period 3, v:[40-60], d:[30-60]"
#
# results = pd.read_csv("nine_periods.csv")
# final = results.shape[0]
# print_figure(final - z, z, name, title)