import math
import matplotlib.pyplot as plt

# Parameters
mean = 0
sigma = 15
start = -100
stop = 100
num = 100

step = (stop - start) / (num - 1)

x = []
pdf = []

for i in range(num):
    xi = start + i * step
    x.append(xi)

    value = (1 / (sigma * math.sqrt(2 * math.pi))) * \
            math.exp(-((xi - mean) ** 2) / (2 * sigma ** 2))
    pdf.append(value)

# Plot
plt.plot(x, pdf)
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Gaussian PDF (mean=0, sigma=15)")
plt.grid(True)
plt.show()