# Define gradient (constant for linear function)
gradient = [2, 3, 3]

# Sample points where we "evaluate" the gradient
points = [
    (0, 0, 0),
    (1, 2, 3),
    (-1, -2, -3),
    (5, 0, -1)
]

print("Gradient of y = 2x1 + 3x2 + 3x3 + 4:\n")

for p in points:
    print(f"At point x1={p[0]}, x2={p[1]}, x3={p[2]} → Gradient = {gradient}")
