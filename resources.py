import random
import logging

def getTrainingSet():
    tSet = []  
    for _ in range(1000):
        n1 = random.randint(0, 100)
        n2 = random.randint(0, 100)
        exp = 0 if n1 < n2 else 1
        tSet.append(([n1, n2], exp))  
    return tSet


def sigmoid(inp):
    try:
        return 1/(1+ pow(2.71828, -inp))
    except Exception as e:
        logging.fatal("Error {e} in sigmoid: input - {inp}!")
        raise
        
def mse(prediction, expRes):
    return 0.5 * pow(prediction-expRes,2)

def step(inp):
    return 1 if inp > 1 else 0