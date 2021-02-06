from NN.neural_net import NeuralNetwork


class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.network = NeuralNetwork(hidden_neurons=3, weights_filename="NN/best.csv")

    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        max_x0 = 668.3179177
        min_x0 = -668.310973

        max_x1 = 589.6073563
        min_x1 = 65.01948654

        x0 = float(input_row.split(",")[0])
        x1 = float(input_row.split(",")[1])

        normalized_x0 = (x0 - min_x0) / (max_x0 - min_x0)
        normalized_x1 = (x1 - min_x1) / (max_x1 - min_x1)

        normalized_predictions = self.network.predict([normalized_x0, normalized_x1])
        
        max_y0 = 8
        min_y0 = -3.684094135

        max_y1 = 7.112776075
        min_y1 = -7.653436076

        prediction0 = normalized_predictions[0] * (max_y0 - min_y0) + min_y0
        prediction1 = normalized_predictions[1] * (max_y1 - min_y1) + min_y1

        return prediction0, prediction1
