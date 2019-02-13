import numpy as np
import matplotlib.pyplot as plt


alpha = 1.0
beta = alpha/4.0
g = 1
tau_z = 4.0
tau_y = 4.0

stime = 100
z = np.zeros(stime)
y = np.zeros(stime)


plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line_y, = ax.plot(list(range(stime)), np.zeros(stime))
line_z, = ax.plot(list(range(stime)), np.zeros(stime))
z_thres = ax.plot(list(range(stime)), np.ones(stime), lw=2)
ax.set_ylim([-.2,1.2])

for t in range(1,stime):

    z[t] = z[t-1] + (1.0/tau_z)*alpha*(beta*(g - y[t-1]) - z[t-1])
    y[t] = y[t-1] + (1.0/tau_y)*z[t-1]

    line_y.set_ydata(y)
    line_z.set_ydata(z)
    fig.canvas.draw()
    plt.pause(0.01)
    print((z[t],y[t]))

