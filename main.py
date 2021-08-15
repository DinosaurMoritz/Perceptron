#pylint:disable=W1201
#pylint:disable=W1203

from Neuron import Neuron
from resources import *
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


    
class NeuralNet:
    
    def __init__(self, initArray=None):
        self.initArray = initArray if initArray != None else []
        
        self.numberOfLayers = len(self.initArray)
        
        self.neurons = []
        inputAmount = self.initArray[0]
        for layerSize in self.initArray:
            self.neurons.append([Neuron(inputAmount) for _ in range(layerSize)])
            inputAmount = len(self.neurons[-1])
    
    def getWeights(self):
        return [[neuron.weights for neuron in layer] for layer in self.neurons]  
    
    @staticmethod
    def calcError(outputs, expected):
        return sum([0.5 * pow(ex - pr, 2) for ex, pr in zip(expected, outputs)])
            
    def predict(self, inputs, function=lambda x: x):
        for layer in self.neurons:
            newInputs = []
            for neuron in layer:
                newInputs.append(neuron.predict(inputs, function))
            
            inputs = newInputs
            
        return inputs
    
    def predictAll(self, inputs, function=lambda x: x):
        everything = []
        for layer in self.neurons:
            newInputs = []
            everythingNew = []
            for neuron in layer:
                prediction = neuron.predict(inputs, function, _all=True)
                newInputs.append(prediction["activation"])
                everythingNew.append(prediction)
              
            inputs = newInputs
            everything.append(everythingNew)
            
        return inputs[0], everything
    
        
        
def main():
    numberOfNeurons = [2,1]
    # trainingExample = [[1], 1]
    net = NeuralNet(numberOfNeurons)
    
    print(net.predict([1,2]))
    
if __name__ == "__main__":
    main()
