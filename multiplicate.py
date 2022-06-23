import sys


def multiplicate(A):
    B = []
    for i in range(len(A)):
        multiplication = 1
        for j in range(len(A)):
            if j != i:
                multiplication *= A[j]
        B.append(multiplication)

    return B


n = len(sys.argv[1])
A = sys.argv[1][1:n - 1]
A = A.split(',')
A = [int(i) for i in A]
print(multiplicate(A))
