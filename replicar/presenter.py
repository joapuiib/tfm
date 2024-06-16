import numpy

class ResultPresenter:
    def __init__(self, cell_width=10):
        self.cell_width = cell_width

    def set_classes(self, classes):
        self.classes = classes

    def set_labels(self, labels):
        self.labels = labels

    def print_header(self,):
        self.print_cells([""], end="")
        self.print_cells(self.labels, cell_witdh=(self.cell_width) * len(self.classes) + (3 * (len(self.classes) - 1)), start="")

        self.print_cells(["Epoch"] + self.classes * len(self.labels));

    def print_cells(self, row, cell_witdh=None, start="|", end="\n"):
        if cell_witdh is None:
            cell_witdh = self.cell_width

        print(start, end="")
        for cell in row:
            if type(cell) in [float, numpy.float64]:
                print(f" {cell:1.{cell_witdh - 2}f} |", end="")
            else:
                print(f" {cell:{cell_witdh}} |", end="")
        print(end, end="")

if __name__ == "__main__":
    presenter = ResultPresenter()
    presenter.set_classes(["A", "B", "C", "D"])
    presenter.set_classes(["A", "B"])
    presenter.set_labels(["Train", "Test"])
    presenter.print_header()
    presenter.print_cells([1] + [0.2, 0.3, 0.4, 0.5]);