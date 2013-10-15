import random
import numpy as np

experiment_size = 1000
sample_size = 10


class SeparatingFunction:
    def __init__(self):
        self.points = np.random.random((2, 2))
        self.line_vector = self.points[0]-self.points[1]

    def test(self, point):
        test_line = point - self.points[1]
        to_test = np.vstack((self.line_vector, test_line))
        if np.linalg.det(to_test) >= 0:
            return 1
        else:
            return -1


class NonLinearFunction:
    def __init__(self):
        def func(X):
            val = np.sign(X[0]**2+X[1]**2 - 0.6)
            if val == 0:
                return 1
            else:
                return val

        self.func = func

    def test(self, point):
        return self.func(point)


class Experiment:
    def run():
        pass


class CoinExpriment(Experiment):
    def run():
        c1 = 0
        cmin = 10
        cRand = 0
        randPick = random.randint(0, 1000)
        for i in range(0, 1000):
            heads = 0
            for j in range(0, 10):
                heads = heads + random.randint(0, 1)
            if i == 0:
                c1 = heads / 10
            if i == randPick:
                cRand = heads / 10
            if heads < cmin:
                cmin = heads
        cmin = cmin / 10
        #print("first coin: ", c1)
        #print("random coin: ", cRand)
        #print("lowest: ", cmin)

        return c1, cRand, cmin


class PointGeneratingExperiment(Experiment):
    def generate_X(self):
        left = np.ones((sample_size, 1))
        right = 2*np.random.random((sample_size, 2)) - 1
        X = np.hstack((left, right))
        return X

    def generate_Y(self, X):
        Y = np.zeros((sample_size, 1))
        for i in range(0, sample_size):
            Y[i] = self.func.test(X[i, 1:])
        return Y


class LinearRegressionClassify(PointGeneratingExperiment):
    function_type = SeparatingFunction

    def __init__(self):
        self.X = self.generate_X()
        self.func = self.function_type()
        self.Y = self.generate_Y(self.X)

    def run(self):
        self.run_regr_class()
        self.run_Eout()

    def run_regr_class(self):
        X = self.X
        Y = self.Y
        XT = X.transpose()
        Xdagger = (np.linalg.inv(XT.dot(X))).dot(XT)
        w = Xdagger.dot(Y)
        g_of_X = np.sign(X.dot(w))
        wrongs = 0
        for i in range(0, sample_size):
            if g_of_X[i] != Y[i]:
                wrongs += 1
        self.w = w
        self.error_in = wrongs/sample_size
        #This works I think but it's not as easy to see why
        #Ein = 1/sample_size * linalg.norm(0.5*(sign(X.dot(w)) - Y))**2
        #return w, Ein

    def run_Eout(self):
        X = self.generate_X()
        Y = self.generate_Y(X)
        w = self.w
        g_of_X = np.sign(X.dot(w))
        wrongs = 0
        for i in range(0, sample_size):
            if g_of_X[i] != Y[i]:
                wrongs += 1
        self.error_out = (wrongs/sample_size)


class LinearRegressionNonLinearClassify(LinearRegressionClassify):
    function_type = NonLinearFunction

    def noisify_Y(self, Y):
        start = random.randint(0, sample_size-1)
        Y[start:start+(.1*sample_size), :] = (
            -1*Y[start:start+(.1*sample_size), :])
        return Y

    def generate_Y(self, X):
        return self.noisify_Y(super().generate_Y(X))


class LinearRegressionNonLinearTransClassify(
        LinearRegressionNonLinearClassify):
    def generate_X(self):
        left = np.ones((sample_size, 1))
        coords = 2*np.random.random((sample_size, 2)) - 1
        coords_mult = np.multiply(coords[:, 0], coords[:, 1])
        coords_left_squared = np.multiply(coords[:, 0], coords[:, 0])
        coords_right_squared = np.multiply(coords[:, 1], coords[:, 1])
        right = np.column_stack((coords_mult, coords_left_squared,
                                 coords_right_squared))
        X = np.hstack((left, coords, right))
        return X


class PerceptronExperiment(PointGeneratingExperiment):
    total_iterations = 0
    converged = False

    def __init__(self, weight=np.zeros((3, 1)), func=None, X=None, Y=None):
        if func is None:
            func = SeparatingFunction()
        self.func = func
        if X is None:
            X = self.generate_X()
            Y = self.generate_Y(X)
        self.X = X
        self.Y = Y
        self.w = weight

    def run(self):
        changed = True
        while changed:
            changed = self.retrain_rand()
            self.total_iterations += 1
        self.converged = True

    def retrain_rand(self):
        changed = False
        X = self.X
        Y = self.Y
        misclassified = []

        for i in range(0, sample_size):
            h_of = np.sign(X[i].dot(self.w))
            if Y[i, 0] != h_of or (h_of == 0 and Y[i, 0] < 0):
                changed = True
                misclassified.append(i)

        if changed:
            i = random.choice(misclassified)
            shift = np.transpose(Y[i, 0]*X[[i]])
            self.w += shift

        return changed


def run_regression_classify():
    Ein = 0
    Eout = 0
    for i in range(0, experiment_size):
        exp = LinearRegressionClassify()
        exp.run()
        Ein += exp.error_in
        Eout += exp.error_out
    print("E_in average: ", Ein/experiment_size)
    print("E_out average: ", Eout/experiment_size)


def run_regression_and_PLA():
    its = 0
    for i in range(0, experiment_size):
        exp = LinearRegressionClassify()
        exp.run()
        perc = PerceptronExperiment(exp.w, None, exp.X, exp.Y)
        perc.run()
        its += perc.total_iterations

    print(its/experiment_size)


def run_non_linear_regression_classify():
    Ein = 0
    Eout = 0
    for i in range(0, experiment_size):
        exp = LinearRegressionNonLinearClassify()
        exp.run()
        Ein += exp.error_in
        Eout += exp.error_out
    print("E_in average: ", Ein/experiment_size)
    print("E_out average: ", Eout/experiment_size)


def run_non_linear_trans_regression_classify():
    Ein = 0
    Eout = 0
    for i in range(0, experiment_size):
        exp = LinearRegressionNonLinearTransClassify()
        exp.run()
        Ein += exp.error_in
        Eout += exp.error_out
    print("E_in average: ", Ein/experiment_size)
    print("E_out average: ", Eout/experiment_size)
    print(exp.w)


def run_hoeffding():
    numOf = 100000
    c1 = 0
    cRand = 0
    cmin = 0
    for i in range(0, numOf):
        exp = CoinExpriment()
        (cur_c1, cur_cRand, cur_cmin) = exp.run()
        c1 = c1 + cur_c1
        cRand = cRand + cur_cRand
        cmin = cmin + cur_cmin
    print(c1/numOf)
    print(cRand/numOf)
    print(cmin/numOf)
