# Memanggil library yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#Memanggil dataset
dataset = pd.read_csv('....csv')
# Sumbu X adalah Gaji, dan Sumbu Y adalah Pengalaman kerja
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
#menampulkan grafik data asli
dataku = pd.DataFrame(dataset)
#Visualisasi Data
plt.scatter(dataku.Tahun, dataku.Gaji)
plt.xlabel("Tahun")
plt.ylabel("Pendapatan")
plt.title("Grafik Masa Kerja vs Pendapatan")
plt.show()