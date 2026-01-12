import matplotlib.pyplot as plt

x1 = []
y = []

for i in range(-100, 101, 2):
    x1.append(i)
    y.append(2 * i + 3)

plt.plot(x1, y)
plt.xlabel("x1")
plt.ylabel("y")
plt.title("y = 2x1 + 3")
plt.grid(True)
plt.show()