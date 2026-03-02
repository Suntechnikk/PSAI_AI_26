import numpy as np
import matplotlib.pyplot as plt

A = np.array([
    [2, 1],
    [-2, 1],
    [2, -1],
    [-2, -1]
], dtype=float)

B = np.array([0, 1, 0, 0], dtype=float)

C = 100
D = 0.01

def E(F):
    return 1 / (1 + np.exp(-F))

def G(H, I):
    J = 1e-9
    return -(I*np.log(H+J) + (1-I)*np.log(1-H+J))

def K(L, M, N):
    O = np.random.randn(2)
    P = np.random.randn()
    Q = []

    for R in range(C):
        S = 0
        for T, U in zip(L, M):
            V = np.dot(O, T) - P
            W = V
            X = U - W
            S += X**2

            O += N * X * T
            P -= N * X

        Q.append(S/len(L))
        if Q[-1] <= D:
            break

    return O, P, Q

def Y(Z, AA):
    O = np.random.randn(2)
    P = np.random.randn()
    Q = []
    AB = 1

    for R in range(C):
        AC = 0
        for T, U in zip(Z, AA):

            AD = 0.5 / AB
            AB += 1

            V = np.dot(O, T) - P
            W = V
            X = U - W
            AC += X**2

            O += AD * X * T
            P -= AD * X

        Q.append(AC/len(Z))
        if Q[-1] <= D:
            break

    return O, P, Q

def AE(AF, AG, AH):
    O = np.random.randn(2)
    P = np.random.randn()
    Q = []

    for R in range(C):
        AI = 0
        for T, U in zip(AF, AG):

            V = np.dot(O, T) - P
            W = E(V)

            AI += G(W, U)

            AJ = (W - U)
            O -= AH * AJ * T
            P += AH * AJ

        Q.append(AI/len(AF))
        if Q[-1] <= D:
            break

    return O, P, Q

def AK(AL, AM):
    O = np.random.randn(2)
    P = np.random.randn()
    Q = []
    AN = 1

    for R in range(C):
        AO = 0
        for T, U in zip(AL, AM):

            AP = 1 / AN
            AN += 1

            V = np.dot(O, T) - P
            W = E(V)

            AO += G(W, U)

            AQ = (W - U)
            O -= AP * AQ * T
            P += AP * AQ

        Q.append(AO/len(AL))
        if Q[-1] <= D:
            break

    return O, P, Q

AR, AS, AT = K(A, B, 0.01)
AU, AV, AW = Y(A, B)
AX, AY, AZ = AE(A, B, 0.01)
BA, BB, BC = AK(A, B)

plt.figure(figsize=(8,5))
plt.plot(AT, color='purple', linewidth=2, label="M1")
plt.plot(AW, color='orange', linewidth=2, label="M2")
plt.plot(AZ, color='green', linewidth=2, label="B1")
plt.plot(BC, color='red', linewidth=2, label="B2")
plt.xlabel("E")
plt.ylabel("V")
plt.title("C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(7,7))

for i in range(len(A)):
    BD = "red" if B[i] == 1 else "blue"
    plt.scatter(A[i,0], A[i,1], color=BD, s=100, edgecolors='black', linewidth=1.5)

BE = np.linspace(-3, 3, 300)
if BA[1] != 0:
    BF = (BB - BA[0]*BE) / BA[1]
    plt.plot(BE, BF, color='darkgreen', linewidth=2.5)

plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid(True, alpha=0.3)
plt.title("D")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

def BG(BH, BI):
    BJ = BA[0]*BH + BA[1]*BI - BB
    BK = E(BJ)
    BL = 1 if BK >= 0.5 else 0


    print(f"P1: {BK:.4f}")
    print(f"C: {BL}")


    plt.figure(figsize=(7,7))

    for i in range(len(A)):
        BM = "red" if B[i] == 1 else "blue"
        plt.scatter(A[i,0], A[i,1], color=BM, s=100, edgecolors='black', linewidth=1.5)

    if BA[1] != 0:
        plt.plot(BE, BF, color='darkgreen', linewidth=2.5)
    plt.scatter(BH, BI, color="magenta", s=200, marker="X", edgecolors='black', linewidth=2, zorder=5)

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True, alpha=0.3)
    plt.title("P")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

while True:
    BN = input("X1 (exit): ")
    if BN.lower() == "exit":
        break
    BO = input("X2: ")

    try:
        BG(float(BN), float(BO))
    except:
        print("E")