# Transpose - Rows to columns and columns to rows

A  = [[1,2,3],
      [4,5,6]]

AT = []

for i in range(len(A[0])):
      row = []
      for j in range(len(A)):
            row.append(A[j][i])
      AT.append(row)

print(AT)

rows = len(AT)
cols = len(A[0])
common = len(A)

ATA = [[0 for _ in range(cols)] for _ in range(rows)]

for i in range(rows):
    for j in range(cols):
        for k in range(common):
            ATA[i][j] += AT[i][k] * A[k][j]

print("A^T X A =")
for row in ATA:
    print(row)