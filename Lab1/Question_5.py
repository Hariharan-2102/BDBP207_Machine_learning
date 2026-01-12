import matplotlib.pyplot as plt

start = -10
stop = 10
num = 100
step = (stop - start) / (num - 1)

x1 = []
y = []

for i in range(num):
    x = start + i * step
    x1.append(x)
    y.append(x**2)

plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y = x1^2")
plt.grid(True)
plt.show()