import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

path_exp = "data/train/20211121-093814"
path_csv = os.path.join(path_exp, "loss.csv")
path_png = os.path.join(path_exp, "loss_figure.png")


csv_file = pd.read_csv(path_csv)
array = csv_file.values
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(array[20:, 0], array[20:, 1])
plt.savefig(path_png)