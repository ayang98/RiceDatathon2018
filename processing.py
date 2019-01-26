import pandas as pd
import numpy as np

df1 = pd.read_csv("pressure.csv")
data1 = df1[['Houston']]
data1 = np.array(data1)

df2 = pd.read_csv("humidity.csv")
data2 = df2[['Houston']]
data2 = np.array(data2)

df3 = pd.read_csv("temperature.csv")
data3 = df3[['Houston']]
data3 = np.array(data3)

df4 = pd.read_csv("wind_speed.csv")
data4 = df4[['Houston']]
data4 = np.array(data4)


final = np.hstack(np.array([data1, data2, data3, data4]))

final = final[1:, :]

print final

