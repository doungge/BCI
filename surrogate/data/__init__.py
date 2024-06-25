
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

class BodyResponseDataset(Dataset):
  def __init__(self, path, classes=15, transform=None):
    df = pd.read_excel(path,engine='openpyxl')
    
    self.transform = transform

    df = df.set_index("index", drop=True)
    
    self.target = [2,3,4,5,6,7,8,11,12,13,14,17,18,19,20]
    if classes == 10:
      self.target = [3,4,5,6,8,12,14,17,18,20]
      df_ = pd.DataFrame()
      for target in self.target:
        df_ = df_.append(df[df["Response"] == target])
      df = df_

    self.dataSize = len(df.index)
    self.targetMap = np.zeros(21)
    for i, target in enumerate(self.target):
      self.targetMap[target] = i
    self.y  = [self.targetMap[i] for i in df["Response"].iloc[:]]
    
    x = df.drop("Response",axis=1)
    mean = x.mean()
    std  = x.std()
    x = (x - mean).div(std)

    self.x = [x.iloc[i].tolist() for i in range(self.dataSize)]
 
  def __len__(self):
    return self.dataSize

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    if isinstance(idx, list):
      size = len(idx)
    elif isinstance(idx, int):
      size = 1
    else:
      size = idx.size
    
    if size > 1:
      x, y = [self.x[i] for i in idx], [self.y[i] for i in idx]
    else:
      x, y = self.x[idx], self.y[idx]
    
    if self.transform:
      x = self.transform(x)
      y = self.transform(y)
    
    return x, y