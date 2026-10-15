# === Importing libraries and modules === #
import pandas as pd
import numpy as np
import scipy.io as sci
import matplotlib.pyplot as plt

# ============= main app ================ #


data = sci.loadmat('/home/ubuntu/BDS_Fault_Detection/data/normal/97.mat')
for key in data.keys():
    if not key.startswith('__'):
        variable = data[key]
        print(f"{key} : type={type(variable).__name__}, shape={getattr(variable, 'shape', 'N/A')}")

df = pd.DataFrame(data['X097_DE_time'])

plt.plot(df)
plt.show()