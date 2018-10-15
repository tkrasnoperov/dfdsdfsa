import numpy as np
from numpy import mean, sign
from numpy.random import permutation, choice, uniform
import matplotlib.pyplot as plt

## PROBLEM 1
def problem_1_2():
    N_EXPERIMENTS = 100000
    N_COINS = 1000
    N_FLIPS = 10

    def flip_coin(n):
        n_heads = sum([choice([0, 1]) for _ in range(n)])
        return n_heads

    def run_experiment(n_exeriments, n_coins, n_flips):
        v_first_list, v_rand_list, v_min_list = [], [], []
        for i in range(n_exeriments):
            coins = [flip_coin(n_flips) for _ in range(n_coins)]
            c_first = coins[0]
            c_rand = choice(coins)
            c_min = min(coins)

            v_first_list.append(c_first / n_flips)
            v_rand_list.append(c_rand / n_flips)
            v_min_list.append(c_min / n_flips)

        print("v_1:", mean(v_first_list))
        print("v_rand:", mean(v_rand_list))
        print("v_min:", mean(v_min_list))

    run_experiment(N_EXPERIMENTS, N_COINS, N_FLIPS)

## PROBLEM 2
def problem_5_6_7():
    N_POINTS = 100
    N_POINTS_PLA = 10
    N_EXPERIMENTS = 1000

    def gen_points(n):
        return uniform(low=-1.0, high=1.0, size=(n,2))

    def gen_target():
        points = gen_points(2)
        a, b = points[0], points[1]

        rise = a[1] - b[1]
        run =  a[0] - b[0]
        slope = rise / run
        intercept = b[1] - b[0] * slope

        f = lambda x: intercept + slope * x
        target = lambda point: 1 if f(point[0]) <= point[1] else -1
        return target

    def get_weights(X, Y):
        # Each vector x must have an additional x_0 dimension as 1 to account
        # for w_0 in w. This will later be removed again for classification.
        faux_X = [[1, x[0], x[1]] for x in X]
        pinv_X = np.linalg.pinv(faux_X)
        return np.dot(pinv_X, Y)

    def classifier_from_weights(w):
        # The weight offset is extracted back out and the X matrix is
        w_0, w = w[0], w[1:]

        g = lambda x: sign(w_0 + sum([w_i * x_i for w_i, x_i in zip(w, x)])) or 1
        return g


    def PLA_convergeance_iters(w, X, Y):
        iters = -1
        has_converged = False
        while not has_converged:
            has_converged = True
            g = classifier_from_weights(w)
            shuffle = permutation(range(len(X)))
            X, Y = X[shuffle], Y[shuffle]

            miss_X = None
            for x, y in zip(X, Y):
                if g(x) != y:
                    x = np.array([1, x[0], x[1]])
                    w = w + y * x
                    has_converged = False
                    break

            iters += 1

        return iters

    def run_5_6(n_experiments, n_points):
        E_in_list = []
        E_out_list = []

        for _ in range(n_experiments):
            f = gen_target()
            X_in = gen_points(n_points)
            X_out = gen_points(n_points)

            Y_in = np.array([f(x) for x in X_in])
            Y_out = np.array([f(x) for x in X_out])

            w = get_weights(X_in, Y_in)
            g = classifier_from_weights(w)

            # # Scatter the actual classification
            # pos_X = [x for x, y in zip(X, Y) if y > 0]
            # neg_X = [x for x, y in zip(X, Y) if y < 0]
            # pos_X_T = np.transpose(pos_X)
            # neg_X_T = np.transpose(neg_X)
            # plt.scatter(pos_X_T[0], pos_X_T[1], color='red')
            # plt.scatter(neg_X_T[0], neg_X_T[1], color='green')
            # plt.show()

            # # Scatter the learned classification
            # pos_X = [x for x, y in zip(X, Y) if g(x) > 0]
            # neg_X = [x for x, y in zip(X, Y) if g(x) < 0]
            # pos_X_T = np.transpose(pos_X)
            # neg_X_T = np.transpose(neg_X)
            # plt.scatter(pos_X_T[0], pos_X_T[1], color='red')
            # plt.scatter(neg_X_T[0], neg_X_T[1], color='green')
            # plt.show()

            n_in_errors = sum([1 for x, y in zip(X_in, Y_in) if g(x) != y])
            n_out_errors = sum([1 for x, y in zip(X_out, Y_out) if g(x) != y])
            E_in = n_in_errors / n_points
            E_out = n_out_errors / n_points
            E_in_list.append(E_in)
            E_out_list.append(E_out)

        print("Problems 5, 6:")
        print("E_in:", mean(E_in_list))
        print("E_out:", mean(E_out_list))

    def run_7(n_experiments, n_points):
        iters_list = []

        for _ in range(n_experiments):
            f = gen_target()
            X = gen_points(n_points)
            Y = np.array([f(x) for x in X])

            w = get_weights(X, Y)
            iters_list.append(PLA_convergeance_iters(w, X, Y))

        print("Problem 7:")
        print("iters:", mean(iters_list))

    run_5_6(N_EXPERIMENTS, N_POINTS)
    print()
    run_7(N_EXPERIMENTS, N_POINTS_PLA)
