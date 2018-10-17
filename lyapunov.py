import numpy as np
import matplotlib.pyplot as plt

seq_len = 100
dt = 1/seq_len
total = 100000

x = np.linspace(0, 2*np.pi, num=total)
'''
y = np.sin(x)
y = np.cos(x)
'''
y = np.random.rand(len(x))

for i in range(int(total/seq_len)):
    l = np.mean(np.log(abs(np.diff(y[i:(i+1)*seq_len-1])/dt)))
    print(l)
    plt.scatter(i, l, c='black', s=1)

plt.show()
