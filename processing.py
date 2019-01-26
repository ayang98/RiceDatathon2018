import pandas as pd
import numpy as np
from collections import Counter 

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


training = np.hstack(np.array([data1, data2, data3, data4]))

training = training[1:, :]

#print final


df5 = pd.read_csv("weather_description.csv")
data5 = df5[['Houston']]
data5 = np.array(data5)

output = data5[1:,:]


list1 = []

for element in output:
	list1.append(str(element))


print len(set(list1))

