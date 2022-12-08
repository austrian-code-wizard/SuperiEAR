import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import struct
from tensorflow.python.summary.summary_iterator import summary_iterator
import os

# import tensorbaord data from ./runs/950Epoch and create a plot


class Chart:
    def __init__(self, data_path, model_name, title, tags, labels, xlabel, ylabel, max_epoch=None):
        self.data_path = data_path
        self.model_name = model_name
        self.title = title
        self.tags = tags
        self.labels = labels
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.max_epoch = max_epoch

    def generate_plot(self):
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        for tag in self.tags:
            values = []
            steps = []
            print(f"Loading data for {tag}")
            for e in summary_iterator(f"{self.data_path}"):
                if self.max_epoch is not None and e.step > self.max_epoch:
                    break
                for v in e.summary.value:
                    if v.tag == tag:
                        values.append(v.simple_value)
                        steps.append(e.step)
            if len(values) == 0:
                raise ValueError(
                    f"Could not find tag {tag} in {self.data_path}")
            plt.plot(steps, values, label=tag)

        plt.legend(self.labels)

        file_name = self.title.replace(" ", "_").lower()
        # make sure the folder exists
        os.makedirs(f"charts-out/{self.model_name}", exist_ok=True)
        plt.savefig(f"charts-out/{self.model_name}/{file_name}.png")

        plt.clf()


DeepConvPath = "./deep-conv-runs/950-epoch/events.out.tfevents.1670494587.ip-172-31-42-40.24931.0"
DeepConvName = "DeepConvAE"

DiffPath = "./diff-runs/events.out.tfevents.1670495421.gpu-train-instance.1024.0"
DiffName = "Diffusion"

if __name__ == "__main__":
    charts = [
        Chart(
            data_path=DeepConvPath,
            model_name=DeepConvName,
            title="Loss per epoch",
            tags=["Loss/epoch", "Loss/val"],
            labels=["Training loss", "Validation loss (combined)"],
            xlabel="Epoch",
            ylabel="Loss"
        ),
        Chart(
            data_path=DeepConvPath,
            model_name=DeepConvName,
            title="Loss per epoch (70 epochs)",
            tags=["Loss/epoch", "Loss/val"],
            labels=["Training loss", "Validation loss (combined)"],
            xlabel="Epoch",
            ylabel="Loss",
            max_epoch=70
        ),
        Chart(
            data_path=DiffPath,
            model_name=DiffName,
            title="Loss per epoch",
            tags=["Loss/epoch", "Loss/val"],
            labels=["Training loss", "Validation loss (combined)"],
            xlabel="Epoch",
            ylabel="Loss"
        ),

        Chart(
            data_path=DeepConvPath,
            model_name=DeepConvName,
            title="Validation loss per epoch",
            tags=["Loss/stfs_total_val", "OG_loss/val", "Loss/train"],
            labels=["STFS loss", "L1 loss", "Combined"],
            xlabel="Epoch",
            ylabel="STFS Loss"
        ),
        Chart(
            data_path=DiffPath,
            model_name=DiffName,
            title="Validation loss per epoch",
            tags=["Loss/stfs_total_val", "OG_loss/val", "Loss/train"],
            labels=["STFS loss", "L1 loss", "Combined"],
            xlabel="Epoch",
            ylabel="STFS Loss"
        ),
    ]
    for chart in charts:
        chart.generate_plot()
