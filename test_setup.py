import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("All imports successful")

# Quick check dataset loads
iris = load_iris()
print("Dataset loaded successfully")
print("Total flowers:", len(iris.data))
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)