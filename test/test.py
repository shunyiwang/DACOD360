#coding:utf8
import os
import pandas as pd  # 引入我们的pandas 模块
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

#rng = np.random.RandomState(seed=12345)
#samples = stats.norm.rvs(size=1000, random_state=rng)
samples = np.zeros(1000)
res = stats.relfreq(samples, numbins=1000)
x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size)
y=np.cumsum(res.frequency)
plt.plot(x,y)
#plt.title('Figure6 累积分布直方图')
plt.show()



