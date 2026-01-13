# <<<<  dot product of two vectors  >>>>

matrix_1 = [20,30,40]
matrix_2 = [50,60,70]

dot_product = 0

for number in range(len(matrix_1)):
    dot_product = dot_product + matrix_1[number] * matrix_2[number]

print(" ------ Dot product ------")
print(dot_product)
