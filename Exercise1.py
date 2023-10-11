import numpy as np


def input_matrix(M, N):
    P = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            while True:
                value = float(input(f"\tEnter P(x{i+1}, y{j+1}): "))
                if value >= 0:
                    P[i, j] = value
                    break
                else:
                    print("Probability cannot be negative. Please re-enter!")
    return P


def entropy(P):
    return -np.sum(P*np.log2(P, where=P != 0))


def joint_entropy(P):
    return entropy(P)


def marginal_probability(P, axis):
    return np.sum(P, axis=axis)


def conditional_entropy(P, axis):
    marginal = marginal_probability(P, 1 - axis)
    return entropy(P) - entropy(marginal)


def mutual_information(P):
    P_x = marginal_probability(P, axis=1)
    P_y = marginal_probability(P, axis=0)
    return entropy(P_x) + entropy(P_y) - joint_entropy(P)


def kullback_leibler_divergence(P, Q):
    return np.sum(P * np.log2(P/Q, where=(P != 0) & (Q != 0)))


def print_matrix(P):
    P_x = marginal_probability(P, axis=1)
    P_y = marginal_probability(P, axis=0)
    print("".center(8, ' '), end=' ')
    for i in range(P.shape[1]):
        print(f"y{i+1}".center(8, ' '), end=' ')
    print("P(x)".center(8, ' '))
    for i in range(P.shape[0]):
        print(f"x{i+1}".center(8, ' '), end=' ')
        for j in range(P.shape[1]):
            print(f"{P[i, j]:.5f}".center(8, ' '), end=' ')
        print(f"{P_x[i]:.5f}".center(8, ' '))
    print("P(y)".center(8, ' '), end=' ')
    for i in range(P.shape[1]):
        print(f"{P_y[i]:.5f}".center(8, ' '), end=' ')
    print()


def main():
    # Sample matrix
    # P = [[0.0625, 0.0625, 0],
    #      [0.375, 0.1875, 0.1875],
    #      [0.0625, 0, 0.0625]]
    # P = np.array(P)

    # Enter the number of rows and columns of the matrix
    M = int(input("Enter number of rows: "))
    N = int(input("Enter number of columns: "))
    P = input_matrix(M, N)
    print(" Input ".center(50, '-'))
    print()
    print_matrix(P)

    # Calculate the entropy, joint entropy, marginal probability, conditional entropy, mutual information, and Kullback-Leibler divergence
    P_x = marginal_probability(P, axis=1)
    P_y = marginal_probability(P, axis=0)
    H_x = entropy(P_x)
    H_y = entropy(P_y)
    H_xy = joint_entropy(P)
    H_y_given_x = conditional_entropy(P, axis=0)
    H_x_given_y = conditional_entropy(P, axis=1)
    MI_xy = mutual_information(P)
    D_Px_Py = kullback_leibler_divergence(P_x, P_y)
    D_Py_Px = kullback_leibler_divergence(P_y, P_x)

    # Print the results
    print()
    print(" Results ".center(50, '-'))
    print()
    print(f"H(X) = {H_x:.5f} bits")
    print(f"H(Y) = {H_y:.5f} bits")
    print(f"H(X,Y) = {H_xy:.5f} bits")
    print(f"H(Y|X) = {H_y_given_x:.5f} bits")
    print(f"H(X|Y) = {H_x_given_y:.5f} bits")
    print(f"H(Y)-H(Y|X) = {H_y-H_y_given_x:.5f} bits")
    print(f"I(X;Y) = {MI_xy:.5f} bits")
    print(f"D(Px||Py) = {D_Px_Py:.5f} bits")
    print(f"D(Py||Px) = {D_Py_Px:.5f} bits")
    print()
    print("-"*50)

if __name__ == "__main__":
    main()
