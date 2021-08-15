# pylint:disable=W1201
# pylint:disable=W1203

from resources import *
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


class Neuron:

    def __init__(self, amountOfInputs, learningRate=0.01):
        self.numberOfWeights = amountOfInputs + 1
        self.weights = [random.random() for _ in range(self.numberOfWeights)]
        self.learningRate = learningRate

    def __repr__(self):
        return f"Neuron {id(self)}"

    def printInfo(self):
        logging.info("----------Info on: " + str(self))
        logging.info("----------Weights: " + str(self.weights))
        logging.info("----------learning rate: " + str(self.learningRate))

    """ activation functions: "step" - binary output/ "sigmoid" - sigmoid activation function"""
    def predict(self, inputsToPredict, func=sigmoid, _all=False):

        if len(inputsToPredict) + 1 != self.numberOfWeights:
            # logging.fatal("Number of inputs to prediction does not equal amount of weights!")
            raise Exception(
                f"Number of inputs to prediction does not equal amount of weights!: inputs - {inputsToPredict}/ weights - {self.weights}")

        output = sum([a * b for a, b in zip(inputsToPredict, self.weights)]) + self.weights[-1]
        activation = func(output)

        if _all:
            return {"neuron": str(self), "activation": activation, "preactivation": output, "weights": self.weights,
                    "inputs": inputsToPredict}
        return activation

    def train(self, trainingSet, iterations=1000, function=sigmoid):
        logging.info(f"Training {iterations} iterations with a set of {len(trainingSet)} examples!")

        weightTemplate = [0 for _ in self.weights]

        for _iteration in range(iterations):
            weights = weightTemplate[:]  # .copy()

            for inputs, expected in trainingSet:
                # logging.debug(f"Training once on: {inputs} and {expected}")
                # logging.debug(f"Old weights: {self.weights}")

                prediction = self.predict(inputs, function)

                error = expected - prediction  # 0.5 * pow(prediction - expRes, 2)

                for i, singleInput in enumerate(inputs):
                    weights[i] += error * singleInput * self.learningRate

                weights[-1] += error * self.learningRate

                # logging.debug(f"new weights: {self.weights}")

            self.weights = [org + n for org, n in zip(self.weights, weights)]

        logging.info(f"Trained to: {self.weights}!")


def main():
    n = Neuron(2)

    n.train(getTrainingSet())

    print()

    for inp, exp in getTrainingSet():
        print(n.predict(inp, step) == exp, inp, n.predict(inp), exp)


if __name__ == "__main__":
    main()
