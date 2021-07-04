import numpy as np
from scipy import stats
from scipy.stats import lognorm
import matplotlib.pyplot as plt

fig = plt.figure()
ax1, ax2 = fig.subplots(1, 2)

def get_lognorm(sigma, mu):
    # s就是sigma，mu
    scale = np.exp(mu)
    # 确定x范围
    x = np.linspace(lognorm.ppf(0.001, sigma), lognorm.ppf(0.990, sigma), 1000)
    # 概率密度函数和随机数
    y_pdf = lognorm.pdf(x, sigma, scale = scale)
    y_rvs = lognorm.rvs(sigma, scale = scale, size = 1000)
    # 输出图像
    ax1.set_title('lognorm')
    ax1.plot(x, y_pdf, 'r-', label = 'lognorm pdf')
    ax1.hist(y_rvs, density = True, histtype = 'stepfilled', alpha = 0.2)

def get_trunc_lognorm(mu, sigma, log_lower, log_upper, data_num=10000):
    norm_lower = np.log(log_lower)
    norm_upper = np.log(log_upper)
    x = np.linspace(log_lower, log_upper, 1000)
    X = stats.truncnorm((norm_lower - mu) / sigma, (norm_upper - mu) / sigma, loc=mu, scale=sigma)
    ax2.plot(x, X.pdf(x))
    norm_data = X.rvs(data_num)
    log_data = np.exp(norm_data)
    return log_data


mu = 1.5
sigma = 0.25
data = get_trunc_lognorm(mu = mu, sigma = sigma, log_lower = 0, log_upper = 20 * 0.27777778)
print(mu)
print(sigma)
print(np.mean(data))
# get_lognorm(mu = mu, sigma = sigma)
# plt.show()
# print(len(data))
# print(data)