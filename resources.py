import random
import logging
import json


def getTrainingSet():
    tSet = []
    for _ in range(100):
        n1 = random.randint(0, 100)
        n2 = random.randint(0, 100)
        exp = 1 if n1 > n2 else 0
        tSet.append(([n1, n2], exp))
    return tSet


def getTrainingSet2():
    tSet = []
    for _ in range(1000):
        n1 = random.randint(0, 100)
        n2 = random.randint(0, 100)
        n3 = random.randint(0, 100)
        exp = 1 if n1 > n2 > n3 else 0
        tSet.append(([n1, n2, n3], exp))
    return tSet


def sigmoid(inp):
    try:
        return 1 / (1 + pow(2.71828, -inp))
    except OverflowError:
        return 1 if inp > 0 else 0


def dSigmoid(inp):
    return sigmoid(inp) * (1 - sigmoid(inp))


def mse(prediction, expRes):
    return 0.5 * pow(prediction - expRes, 2)


def step(inp, offset=0.001):
    return 1 if inp > offset else 0


def formatJson(inp):
    inp = (str(inp)).replace("\'", "\"")
    json_object = json.loads(inp)
    return json.dumps(json_object, indent=2)

