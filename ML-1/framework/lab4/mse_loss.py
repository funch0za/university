from layer import Layer

class MSELoss(Layer):
    def forward(self, prediction, true_prediction):
        square_diff = (prediction - true_prediction) * (prediction - true_prediction)
        return square_diff.__sum__(0)
