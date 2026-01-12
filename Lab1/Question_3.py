import matplotlib.pyplot as plt

x1 = []
y = []

start = -10
stop = 10
num = 100
step = (stop - start) / (num - 1)

for i in range(num):
    x = start + i * step
    x1.append(x)
    y.append(2 * x**2 + 3*x + 4)

plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y = 2x1^2 + 3x1 + 4")
plt.grid(True)
plt.show()