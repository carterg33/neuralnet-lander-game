import random
import math

def dot(a, b):
  return sum(x * y for x, y in zip(a, b))

class Neuron:
  def __init__(self):
    self.connections = {}
    self.error = 0
    self.previous_weight_delta = 0
  
  #initialise neuron weights
  def connect(self, layer, weights=None):
    for neuron in layer:
      if weights:
        self.connections[neuron] = weights.pop(0)
      else:
        self.connections[neuron] = random.uniform(0, 1)

  def sigmoid(self, s):
    return 1/(1 + math.exp(-s))

  def feed_forward(self):
    self.input_values = []
    weights = []
    for neuron, weight in self.connections.items():
      self.input_values.append(neuron.value)
      weights.append(weight)
    value = dot(weights, self.input_values)
    self.value = self.sigmoid(value)
    return self.value

class NeuralNetwork:
  def __init__(self, hidden_neurons, weights_filename = None):
    # initiatlise neurons
    self.input_layer = [Neuron(), Neuron()]
    self.output_layer = [Neuron(), Neuron()]
    self.hidden_layer = []
    for _ in range(hidden_neurons):
      self.hidden_layer.append(Neuron())

    # create connections
    weights = None
    if weights_filename:
      weights = self.load_weights(weights_filename)
    for neuron in self.hidden_layer:
      neuron.connect(self.input_layer, weights)
    for neuron in self.output_layer:
      neuron.connect(self.hidden_layer, weights)

  def transfer_derivative(self, value):
    return value * (1 - value)
  
  def predict(self, inputs):
    self.input_layer[0].value = inputs[0]
    self.input_layer[1].value = inputs[1]
    for neuron in self.hidden_layer:
      neuron.feed_forward()
    outputs = []
    for neuron in self.output_layer:
      outputs.append(neuron.feed_forward())
    return outputs

  def train(self, inputs, outputs, test_inputs, test_outputs):
    self.learn_rate = 0.3
    self.momentum = 0.9
    epochs = 100

    previous_train_rmse = 1
    previous_test_rmse = 1

    for epoch in range(epochs):
      for X, y in zip(inputs, outputs):
        # feed in data to input layer
        self.input_layer[0].value = X[0]
        self.input_layer[1].value = X[1]
        # feed forward
        for neuron in self.hidden_layer:
          neuron.feed_forward()
          neuron.error = 0
        results = []
        for neuron in self.output_layer:
          results.append(neuron.feed_forward())
        
        # backpropagate errors
        for index, neuron in enumerate(self.output_layer):
          neuron.error = (y[index] - neuron.value) * self.transfer_derivative(neuron.value)
          for connection, weight in neuron.connections.items():
            connection.error += (neuron.error * weight) * self.transfer_derivative(connection.value)

        # update weights
        for neuron in self.output_layer:
          for connection, weight in neuron.connections.items():
            weight_delta = (self.learn_rate * neuron.error * connection.value) + (self.momentum * neuron.previous_weight_delta)
            neuron.connections[connection] += weight_delta
            neuron.previous_weight_delta = weight_delta
          
        for neuron in self.hidden_layer:
          for connection, weight in neuron.connections.items():
            weight_delta = (self.learn_rate * neuron.error * connection.value) + (self.momentum * neuron.previous_weight_delta)
            neuron.connections[connection] += weight_delta
            neuron.previous_weight_delta = weight_delta

      train_rmse = self.rmse_loss(inputs, outputs)
      test_rmse = self.rmse_loss(test_inputs, test_outputs)

      if train_rmse > previous_train_rmse or test_rmse > previous_train_rmse:
        print("Early stop at epoch %d train loss: %.6f, test loss: %.6f" % (epoch, train_rmse, test_rmse))
        break
      
      previous_train_rmse = train_rmse
      previous_test_rmse = test_rmse
      
      # calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        print("Epoch %d train loss: %.6f, test loss: %.6f" % (epoch, train_rmse, test_rmse))

  def rmse_loss(self, inputs, outputs):
    #add rmse
    first_errors = []
    second_errors = []
    for X, y in zip(inputs, outputs):
      predicitions = self.predict(X)
      
      first_errors.append((y[0] - predicitions[0]) ** 2)
      second_errors.append((y[1] - predicitions[1]) ** 2)
    first_rmse = math.sqrt(sum(first_errors) / len(first_errors))
    second_rmse = math.sqrt(sum(second_errors) / len(second_errors))

    return (first_rmse + second_rmse) / 2
  
  def save_weights(self, weights_filename):
    weights = []

    for neuron in self.hidden_layer:
      for connection, weight in neuron.connections.items():
        weights.append(weight)

    for neuron in self.output_layer:
      for connection, weight in neuron.connections.items():
        weights.append(weight)
      
    with open(weights_filename, "w") as weights_file:
      weights_file.write(str(weights).strip("[]"))
  
  def load_weights(self, weights_filename):
    with open(weights_filename) as weights_file:
      weights_string = weights_file.readline().strip().split(",")
    weights = [float(weight) for weight in weights_string]
    return weights

if __name__ == "__main__":
  inputs = []
  with open("NN/X_train.csv") as inputs_file:
    for line in inputs_file:
      first_input = float(line.strip().split(",")[0])
      second_input = float(line.strip().split(",")[1])
      inputs.append([first_input, second_input])

  outputs = []
  with open("NN/y_train.csv") as outputs_file:
    for line in outputs_file:
      first_output = float(line.strip().split(",")[0])
      second_output = float(line.strip().split(",")[1])
      outputs.append([first_output, second_output])

  test_inputs = []
  with open("NN/X_test.csv") as inputs_file:
    for line in inputs_file:
      first_input = float(line.strip().split(",")[0])
      second_input = float(line.strip().split(",")[1])
      test_inputs.append([first_input, second_input])

  test_outputs = []
  with open("NN/y_test.csv") as outputs_file:
    for line in outputs_file:
      first_output = float(line.strip().split(",")[0])
      second_output = float(line.strip().split(",")[1])
      test_outputs.append([first_output, second_output])

  # train neural network
  network = NeuralNetwork(hidden_neurons=3)
  network.train(inputs, outputs, test_inputs, test_outputs)
  network.save_weights("NN/weights.csv")
