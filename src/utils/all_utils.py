"""
author : Vishal

"""


import numpy as np
import os
import matplotlib.pyplot as plt
import joblib
from matplotlib.colors import ListedColormap
import logging

plt.style.use("fivethirtyeight")


def prepare_data(df):
  """it is used to separate the dependent and independent variables

  Args:
      df (pd.DataFrame): its Pandas DataFrame which stores dataset

  Returns:
      tuple: it returns the tuple of dependent and independent features
  """
  logging.info("Preparing data by separating independent and dependent features")
  x = df.drop("y",axis=1)
  y= df["y"]
  return x,y


def save_model(model,filename):
  """This saves the trained model

  Args:
      model (python object): trained model
      filename (str): path to save the trained model
  """
  logging.info("Saving the trained model")
  model_dir='models'
  os.makedirs(model_dir,exist_ok=True)
  filepath = os.path.join(model_dir,filename)
  joblib.dump(model,filepath)
  logging.info(f"saved the trained model {filepath}")

def save_plot(df, file_name, model):
  """This is used to create base plot and decison boundry and save the plot file

  Args:
      df (pd.DataFrame): its Pandas DataFrame which stores dataset
      file_name (str): path to save the plot
      model (python object): trained model
  """

  def _create_base_plot(df): ## local function for creating base plot
    logging.info("Creating base plot")
    df.plot(kind="scatter", x="x1", y="x2", c="y", s=100, cmap="winter")
    plt.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
    figure = plt.gcf() # get current figure
    figure.set_size_inches(10, 8)

  def _plot_decision_regions(X, y, classfier, resolution=0.02): ## local function for creating decision boundry
    logging.info("Plotting the decision regions")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    X = X.values # as a array
    x1 = X[:, 0] 
    x2 = X[:, 1]
    x1_min, x1_max = x1.min() -1 , x1.max() + 1
    x2_min, x2_max = x2.min() -1 , x2.max() + 1  

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min, x2_max, resolution))
    
    Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    plt.plot();



  X, y = prepare_data(df)

  _create_base_plot(df)
  _plot_decision_regions(X, y, model)

  plot_dir = "plots"
  os.makedirs(plot_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  plotPath = os.path.join(plot_dir, file_name) # model/filename
  plt.savefig(plotPath)
  logging.info(f"Saved the plot {plotPath}")
