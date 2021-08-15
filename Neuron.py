#pylint:disable=W1201
#pylint:disable=W1203

import time
from resources import *
import logging 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')



class Neuron:
    
    def __init__(self, _amountOfInputs, _learningRate=0.005):
        self.numberOfWeights = _amountOfInputs + 1
        self.weights = [random.random() for _ in range(self.numberOfWeights)]
        self.learningRate = _learningRate
  
    
    def __repr__(self):
        return f"Neuron {id(self)}"
        
    def printInfo(self):
        logging.info("----------Info on: "+str(self))
        logging.info("----------Weights: "+ str(self.weights))
    
    
    # activation functions: "step" - binary output/ "sigmoid" - sigmoid activation function      
    def predict(self, inputsToPredict, func=lambda x: x, _all=False):
        #print(inputsToPredict)
        inputsPlusOne = inputsToPredict + [1]
        if len(inputsPlusOne) != self.numberOfWeights:
            logging.fatal("Number of inputs to prediction does not equal amount of weights!")
            raise Exception(f"Number of inputs to prediction does not equal amount of weights!: inputs - {inputsToPredict}/ weights - {self.weights}")
        
        output = sum([a*b for a, b in zip(inputsPlusOne, self.weights)])
        
        if _all:
            return {"neuron":str(self),"activation":func(output), "preactivation":output, "weights":self.weights, "inputs":inputsToPredict}
        return func(output)
    
    
    def trainOnce(self, inputs, expected, function=lambda x: x):
        logging.debug(f"Training once on: {inputs} and {expected}")
        logging.debug(f"Old weights: {self.weights}")
      
        prediction = self.predict(inputs)
        
        print("prediction", prediction)
        
        error = expected - prediction
        
        for i, singleInput in enumerate(inputs):
            self.weights[i] += error * singleInput * self.learningRate  
        
        self.weights[-1] += error * self.learningRate
        
        logging.debug(f"new weights: {self.weights}")
        time.sleep(0.1)
    
    
    def train(self, trainingSet, iterations = 100):
        logging.info(f"Training {iterations} iterations with a set of {len(trainingSet)} examples!")
        
        for _iteration in range(iterations):
            for inputs, expected in trainingSet:
                self.trainOnce(inputs, expected)
                
        logging.info(f"Trained to: {self.weights}!")
        
        
def main():
    n = Neuron(2)
    #print(n.predict([10,230]))
    #print(getTrainingSet())
    n.train(getTrainingSet())
    
    #print(n.predict([10,20,30]))
    
    #exit()
    #for inp, ex in getTrainingSet2():
    #    print(n.predict(inp))
    
    #n.weights = [-0.10759003168915993, 0.05022616030864452, 0.23165798285560857]

    #print(n.predict([5,5]))
    #[print(x) for x in getTrainingSet2()]
    # n.trainOnceNew(([6],1)) #(getTrainingSet()) #Once([1,0], 0)
    
    #print(n.weights)#predict([40, 20], step))
    
    #n.weights = [11.822518621579214, -11.741960936794467, 0.5260794824047347]
    #for inp, exp in getTrainingSet():
    #    print(n.predict(inp, step) == exp)
    
    
if __name__ == "__main__":
    main()