#pylint:disable=W1201
#pylint:disable=W1203

from resources import *
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')




class Neuron:
    
    def __init__(self, _amountOfInputs, _learningRate=0.01):
        self.numberOfWeights = _amountOfInputs + 1
        self.weights = [random.random() for _ in range(self.numberOfWeights)]
        self.learningRate = _learningRate
  
    
    def __repr__(self):
        return f"Neuron {id(self)}"
        
    def printInfo(self):
        logging.info("----------Info on: "+str(self))
        logging.info("----------Weights: "+ str(self.weights))
    
    
    # activation functions: "step" - binary output/ "sigmoid" - sigmoid activation function      
    def predict(self, inputsToPredict, func=sigmoid, _all=False):
        #print(inputsToPredict)
       
        if len(inputsToPredict) + 1 != self.numberOfWeights:
            logging.fatal("Number of inputs to prediction does not equal amount of weights!")
            raise Exception(f"Number of inputs to prediction does not equal amount of weights!: inputs - {inputsToPredict}/ weights - {self.weights}")
        
        output = sum([a*b for a, b in zip(inputsToPredict, self.weights)]) + self.weights[-1]
        activation = func(output)
 
        if _all:
            return {"neuron":str(self),"activation":activation, "preactivation":output, "weights":self.weights, "inputs":inputsToPredict}
        return activation
    
    
    def trainOnce(self, inputs, expected, function=sigmoid):
        logging.debug(f"Training once on: {inputs} and {expected}")
        logging.debug(f"Old weights: {self.weights}")
      
        prediction = self.predict(inputs, function)
        
        error = expected - prediction
        
        #weights = self.weights
        
        for i, singleInput in enumerate(inputs):
            self.weights[i] += error * singleInput * self.learningRate  
        
        self.weights[-1] += error * self.learningRate
        
        
        logging.debug(f"new weights: {self.weights}")
       
        #return weights
        
    
    def train(self, trainingSet, iterations = 100):
        logging.info(f"Training {iterations} iterations with a set of {len(trainingSet)} examples!")
        
        #lenTrainingSet = len(trainingSet)
        #lenInputs = len(trainingSet[0][0])
        
        for _iteration in range(iterations):
            logging.info(f"Training iteration {_iteration}!")
            
            #weightsSumOfExamples = [0 for _ in range(lenInputs)]
            
            for inputs, expected in trainingSet:
                self.trainOnce(inputs, expected)
                
                #weightsSumOfExamples = [ex + sugg for ex, sugg in zip(weightsSumOfExamples, weightsSuggestion)]
                
            #self.weights = [x/lenTrainingSet for x in weightsSumOfExamples] 
             
        logging.info(f"Trained to: {self.weights}!")


def main():
    n = Neuron(3)
    
    #print(getTrainingSet())
    
    n.weights = [6.851003718620061, -7.249199686306951, -0.07176739849979419, -2.577605420457377]
    
    #n.train(getTrainingSet2()) #Once([1,0], 0)
    
    #print(n.predict([5,10,15]))
    
    #print(n.weights)#predict([40, 20], step))
    
    #n.weights = [-3.66201652,1.335772078, 2.7347970488, -132.63985473891]

    
    
    for inp, exp in getTrainingSet2():
 
        print(n.predict(inp,step) == exp, inp, round(n.predict(inp), 5), exp)
        
 
    
if __name__ == "__main__":
    main()