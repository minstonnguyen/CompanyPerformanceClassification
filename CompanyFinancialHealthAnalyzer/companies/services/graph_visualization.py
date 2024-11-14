import matplotlib.pyplot as plt

class GraphVisualization:
    #xvals and yVals should be of type DataFrame or of a list of some sort
    def __init__(self, graphTitle, xAxisValues, yAxisValues, xAxisLabel,yAxisLabel):
        self.graphTitle = graphTitle
        self.xAxisValues = xAxisValues
        self.yAxisValues = yAxisValues
        self.xAxisLabel = xAxisLabel
        self.yAxisLabel = yAxisLabel
    def plot_kpi_distribution(self):
        plt.plot(self.xAxisValues, self.yAxisValues)
        plt.title(self.graphTitle)
        plt.xlabel(self.xAxisLabel)
        plt.ylabel(self.yAxisLabel)
        plt.grid(True)
        plt.legend(loc='upper left')
    def show_plot(self):
        plt.show()

