import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)

mu, sigma = 100, 15
# x = mu + sigma * np.random.randn(10000)
data = np.load('std_devs.npy')

# the histogram of the data
x = data[:, :, 0]
plt.figure()
n, bins, patches = plt.hist(x, 1000, density=True, facecolor='r', alpha=0.75)
plt.xlabel('Std Dev')
plt.ylabel('Count')
plt.title('Histogram')
plt.grid(True)
plt.show()

x = data[:, :, 1]
plt.figure()
n, bins, patches = plt.hist(x, 1000, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Std Dev')
plt.ylabel('Count')
plt.title('Histogram')
plt.grid(True)
plt.show()

x = data[:, :, 2]
plt.figure()
n, bins, patches = plt.hist(x, 1000, density=True, facecolor='b', alpha=0.75)
plt.xlabel('Std Dev')
plt.ylabel('Count')
plt.title('Histogram')
plt.grid(True)
plt.show()
