import random


class SeparatingFunction:
    def __init__(self):
        self.a = random_point()
        self.b = random_point()
        self.line_vector = vector_subtract(self.a, self.b)

    def test(self, point):
        test_line = vector_subtract(point, self.b)
        return cross_product(self.line_vector, test_line) >= 0


def cross_product(a, b):
    (a0, a1, a2) = a
    (b0, b1, b2) = b
    return (a1*b2) - (b1*a2)


def vector_add(a, b):
    (a1, a2, a3) = a
    (b1, b2, b3) = b
    return (a1+b1, a2+b2, a3+b3)


def inner_product(a, b):
    (a1, a2, a3) = a
    (b1, b2, b3) = b
    return (a1*b1+a2*b2+a3*b3)


def vector_subtract(a, b):
    (a1, a2, a3) = a
    (b1, b2, b3) = b
    return (a1-b1, a2-b2, a3-b3)


def random_point():
    point = (1, random.uniform(-1, 1), random.uniform(-1, 1))
    return point


def retrain_rand(points, weight):
    changed = False
    misclassified = []
    for point, y in points:
        if weight_value(weight, point) is not y:
            misclassified.append((point, y))

    for point, y in random.shuffle(misclassified):
        changed = True
        global iterations
        iterations += 1
        if y:
            weight = vector_add(weight, point)
        else:
            weight = vector_subtract(weight, point)

    return changed, weight


def retrain(points, weight):
    changed = False
    for point, y in points:
        if weight_value(weight, point) is not y:
            changed = True
            global iterations
            iterations += 1
            if y:
                weight = vector_add(weight, point)
            else:
                weight = vector_subtract(weight, point)

    return changed, weight


def main(weight=(0, 0, 0), retrain=retrain, function = SeparatingFunction(), points=[]):
    if points == []:
        for i in range(0, 10):
            new_point = random_point()
            points.append((new_point, function.test(new_point)))
 
 changed = True
    while changed:
        (changed, weight) = retrain(points, weight)
    return (weight, function)

iterations = 0

equality_fails = 0


def check_function(weight, func):
    point = random_point()
    real_value = func.test(point)
    if real_value is not weight_value(weight, point):
        global equality_fails
        equality_fails += 1


def weight_value(weight, point):
    return inner_product(weight, point) >= 0

