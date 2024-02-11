import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import axes3d


class GradientDescent:
    def __init__(self):
        self.data = pd.read_csv(
            r"/Users/neverland/Documents/AIProject/src/irisdata.csv"
        )
        self.learning_rate = 0.1
        self.iterations = 100
        self.weights = np.random.rand(self.data.shape[1] - 1)  # Initialize weights

    def gradient_descent(
        self, data, target, initial_weights, initial_bias, learning_rate, iterations
    ):
        # Initialize weights and bias
        weights = initial_weights
        bias = initial_bias

        objective_function_values = []  # To store objective function values

        # Perform gradient descent
        for i in range(iterations):
            predictions = self.onelayer_neural_network(data, weights, bias)

            # Calculate gradients
            weight_gradient, bias_gradient = self.sum_gradient(
                weights, bias, data, target
            )

            # Update weights and bias
            weights -= learning_rate * weight_gradient
            bias -= learning_rate * bias_gradient

            # Compute objective function value (Mean Squared Error)
            mse = self.mean_squared_error(data, weights, bias, target)
            objective_function_values.append(mse)

            # Plot decision boundary for specific iterations
            if i in [0, iterations // 2, iterations - 1]:
                self.plot_decision_boundary(weights, bias)

        self.plot_learning_curve(objective_function_values)

        # Plot final decision boundary after iterations
        self.plot_decision_boundary(weights, bias)

    def plot_learning_curve(self, objective_function_values):
        plt.figure(figsize=(8, 6))
        iterations = range(1, len(objective_function_values) + 1)
        plt.plot(iterations, objective_function_values)
        plt.title("Learning Curve: Objective Function vs Iterations")
        plt.xlabel("Iterations")
        plt.ylabel("Objective Function (MSE)")
        plt.show()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def filter_class2_3(self):
        self.class_2_3 = self.data[
            self.data["species"].isin(["versicolor", "virginica"])
        ]

        petal_width_array = self.class_2_3["petal_width"].to_numpy()
        petal_length_array = self.class_2_3["petal_length"].to_numpy()
        species_array = self.class_2_3["species"].to_numpy()

        return petal_width_array, petal_length_array, species_array

    # issue of the function need to check this again
    def onelayer_neural_network(self, input, weights, bias):
        weight_sum = np.dot(input, weights) + bias
        prediction = self.sigmoid(weight_sum)
        prediction_collect = []
        for i in range(0, len(prediction), 1):
            if prediction[i] > 0.5:
                prediction_collect.append(1)
            else:
                prediction_collect.append(0)
        return np.array(prediction_collect)

    def plot_decision_boundary(self, weights, bias):
        intercept = -(bias / weights[1])
        slope = -(weights[0] / weights[1])

        x = np.array([2, 7])
        y = slope * x + intercept
        species_color = {"versicolor": "blue", "virginica": "green"}

        figure = go.Figure()
        petal_width, petal_length, species = self.filter_class2_3()

        figure.add_trace(
            go.Scatter(
                x=petal_length,
                y=petal_width,
                mode="markers",
                name="points",
                marker=dict(color=[species_color[s] for s in species]),
            )
        )
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Decision Boundary"))
        figure.update_layout(title="Linear Decision Boundary")
        figure.update_xaxes(title_text="Petal Length")
        figure.update_yaxes(title_text="Petal Width")
        figure.show()

    # pattern class = the optimal class
    def mean_squared_error(self, data, weights, bias, pattern_classes):
        outputs = self.onelayer_neural_network(data, weights, bias)
        squared_errors = (outputs - pattern_classes) ** 2
        return np.mean(squared_errors)

    # classes = defines what class
    def sum_gradient(self, weights, bias, data, target):
        gradient = np.zeros_like(weights)
        bias_gradient = 0

        predictions = self.onelayer_neural_network(data, weights, bias)

        # Calculate bias gradient
        bias_gradient = np.sum(predictions - target)

        # Calculate gradients for weights
        for i in range(len(weights)):
            gradient_sum = np.sum((predictions - target) * data[:, i])
            gradient[i] = gradient_sum

        return gradient, bias_gradient

    def decision_boundary_converge(
        self, start_weights, start_bias, data, target, step_size, iter
    ):
        weights = start_weights
        bias = start_bias

        for i in range(iter):
            predictions = self.onelayer_neural_network(data, weights, bias)
            self.plot_decision_boundary(weights, bias)

            weight_gradient, bias_gradient = self.sum_gradient(
                weights, bias, data, target
            )

            # Update weights and bias
            weights -= step_size * weight_gradient
            bias -= step_size * bias_gradient

        # Plot final decision boundary after iterations
        self.plot_decision_boundary(weights, bias)
