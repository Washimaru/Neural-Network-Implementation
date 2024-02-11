import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import axes3d


class DataPlot:
    def __init__(self):
        self.data = pd.read_csv(
            r"/Users/neverland/Documents/AIProject/src/irisdata.csv"
        )
        self.class_2_3 = None

    def filter_class2_3(self):
        self.class_2_3 = self.data[
            self.data["species"].isin(["versicolor", "virginica"])
        ]

        petal_width_array = self.class_2_3["petal_width"].to_numpy()
        petal_length_array = self.class_2_3["petal_length"].to_numpy()
        species_array = self.class_2_3["species"].to_numpy()

        return petal_width_array, petal_length_array, species_array

    def graph_data(self, petal_width, petal_length, species_array, xsize, ysize):
        plt.figure(figsize=(xsize, ysize))
        petal_width, petal_length, species = self.filter_class2_3()

        species_color = {"versicolor": "blue", "virginica": "green"}

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=petal_length,
                y=petal_width,
                mode="markers",
                name="points",
                marker=dict(color=[species_color[s] for s in species]),
            )
        )
        figure.update_xaxes(title_text="Petal Length")
        figure.update_yaxes(title_text="Petal Width")
        figure.update_layout(title="Linear Decision Boundary")
        figure.show()

    # sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

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

        x = np.linspace(3, 7, 100)
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

    def surface_plot(self, weights, bias):
        petal_width_vals = np.linspace(
            min(self.data["petal_width"]), max(self.data["petal_width"]), 100
        )
        petal_length_vals = np.linspace(
            min(self.data["petal_length"]), max(self.data["petal_length"]), 100
        )
        petal_width_grid, petal_length_grid = np.meshgrid(
            petal_width_vals, petal_length_vals
        )

        inputData = np.c_[petal_length_grid.ravel(), petal_width_grid.ravel()]
        outputData = self.onelayer_neural_network(inputData, weights, bias).reshape(
            petal_length_grid.shape
        )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(petal_width_grid, petal_length_grid, outputData, cmap="viridis")

        ax.set_xlabel("Petal Length")
        ax.set_ylabel("Petal Width")
        ax.set_zlabel("Neural Network Output")
        ax.set_title("Output of One Layer Neural Network over Input Space")

        plt.show()
