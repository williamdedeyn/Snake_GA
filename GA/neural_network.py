import numpy as np
import tkinter as tk

sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
linear = lambda x: x
relu = lambda x: np.maximum(0,x)


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.parameters = {}
        self.init_weights()
        self.output_activation = sigmoid
        self.hidden_activation = relu

    def init_weights(self):
        for i in range(0, len(self.layers) - 1):
            self.parameters['W' + str(i)] = np.random.uniform(-1, 1, size=(self.layers[i + 1], self.layers[i]))
            self.parameters['B' + str(i)] = np.random.uniform(-1, 1, size=(self.layers[i + 1], 1))

    def feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        if len(inputs.shape) != 2:
            raise Exception("Make sure the input of the neural network is iof shape (#inputs,1)")
        temp = inputs
        self.parameters['L' + str(0)] = inputs
        for i in range(0, len(self.layers) - 1):
            weights = self.parameters.get('W' + str(i))
            bias = self.parameters.get('B' + str(i))
            z = np.dot(weights, temp) + bias
            if i == len(self.layers)-2:
                temp = self.output_activation(z)
            else:
                temp = self.hidden_activation(z)
            self.parameters['L' + str(i+1)] = temp
        return temp

    def set_parameters(self,parameters):
        self.parameters = parameters


class NeuralNetworkViz():
    def __init__(self, layers):
        self.layers = layers
        self.window = tk.Tk()
        self.window.title('Neural Network Visualisation')
        self.size_neurons = 20
        self.distance_neurons = 180
        self.padding = 10
        self.width = 960
        self.height = 960
        self.number_layers = len(self.layers)
        self.x = 40
        self.y = 300
        self.neuron_coor = {}
        self.generate_canvas()

    def generate_canvas(self):
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height, bg="white")
        self.canvas.pack()

    def draw(self,parameters):
        self.canvas.delete('all')
        self.x = self.width / 2 - (
                self.number_layers * (self.size_neurons + self.distance_neurons) - self.distance_neurons) / 2
        for index, nn_neurons in enumerate(self.layers):
            self.y = self.height / 2 - (nn_neurons * (self.size_neurons + self.padding) - self.padding) / 2
            for i in range(0, nn_neurons):
                color_neuron = 'white'
                neuron_value = parameters.get('L'+ str(index))[i]
                if index == 0 and neuron_value > 0:
                    color_neuron = '#90ee90'
                elif index == len(self.layers)-1:
                    max_output = np.argmax(parameters.get('L' + str(index)))
                    if i == max_output:
                        color_neuron = '#00BFFF'
                elif (neuron_value > 0.5):
                    color_neuron = '#90ee90'
                self.canvas.create_oval(self.x, self.y,
                                        self.x + self.size_neurons, self.y + self.size_neurons, outline="black",
                                        fill=color_neuron)
                self.neuron_coor[(index,i)] = (self.x,self.y)
                self.y += self.size_neurons + self.padding
            self.x += self.distance_neurons
        for layer in range(0, len(self.layers)-1):
            weights = parameters.get('W' + str(layer))
            for index_output in range(0,len(weights)): # iterate over rows
                for index_input in range(0,len(weights[0])): # iterate over columns
                    value_weight = weights[index_output][index_input]
                    if value_weight > 0:
                        color = "blue"
                    else:
                        color = 'red'
                    start_node = self.neuron_coor.get((layer,index_input))
                    end_node = self.neuron_coor.get((layer+1, index_output))
                    self.canvas.create_line(start_node[0] + self.size_neurons,
                                            start_node[1] + self.size_neurons/2,
                                            end_node[0],
                                            end_node[1] + self.size_neurons/2,
                                    fill=color)
        self.window.mainloop()

    def close(self):
        self.window.destroy()