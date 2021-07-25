#pylint:disable=W1201
#pylint:disable=W1203

from resources import *
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


class Neuron:
    
    def __init__(self, _amountOfInputs, _learningRate=0.05, _bias=0):
        self.numberOfWeights = _amountOfInputs + 1
        self.weights = [random.random() for _ in range(self.numberOfWeights)]
        self.learningRate = _learningRate
        self.bias = _bias
    
    
    def printInfo(self):
        logging.info("----------Info on: "+str(self))
        logging.info("----------Weights: "+ str(self.weights))
    
    # activation functions: "step" - binary output/ "sigmoid" - sigmoid activation function      
    def predict(self, inputsToPredict, func=lambda x: x):
        inputsPlusOne = inputsToPredict + [1]
        if len(inputsPlusOne) != self.numberOfWeights:
            logging.fatal(f"Number of inputs to prediction does not equal amount of weights!: inputs - {inputsToPredict}/ weights - {self.weights}")
            raise Exception(f"Number of inputs to prediction does not equal amount of weights!: inputs - {inputsToPredict}/ weights - {self.weights}")
        
        output = sum([a*b for a, b in zip(inputsPlusOne, self.weights)]) + self.bias
        return func(output)
    
    
    def trainOnce(self, inputs, expected, function=step):
        logging.debug(f"Training once on: {inputs} and {expected}")
        logging.debug(f"Old weights: {self.weights}")
      
        prediction = self.predict(inputs, function)
        
        error = expected - prediction
        
        for i, singleInput in enumerate(inputs):
            self.weights[i] += error * singleInput * self.learningRate  
        
        self.weights[-1] += error * self.learningRate
        
        logging.debug(f"new weights: {self.weights}")
    
    
    def train(self, trainingSet, iterations = 100):
        logging.info(f"Training {iterations} iterations with a set of {len(trainingSet)} examples!")
        
        for _iteration in range(iterations):
            for inputs, expected in trainingSet:
                self.trainOnce(inputs, expected)
                
        logging.info(f"Trained to: {self.weights}!")




def main():
    n = Neuron(2)
    
    n.train(getTrainingSet()) #Once([1,0], 0)
     
    # print(n.predict([50, 40],step))
    
    for i,x in getTrainingSet():
        print(x, n.predict(i, step))
        
if __name__ == "__main__":
    main()