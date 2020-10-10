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

#Memecah data training dan data testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#Fitting / mempersiapkan metode regresi linear
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Visualisasi Training process
plt.figure(figsize=(10,8))
#Biru adalah data observasi
plt.scatter(x_train, y_train, color='blue')
#Garis Merah adalah hasil prediksi dari machine learning
plt.plot(x_train, regressor.predict(x_train), color = 'red')
plt.title('Masa Kerja terhadap pendapatan')
plt.xlabel('Masa kerja')
plt.ylabel('Pendapatan')
plt.show()

#Visualisasi Testing Process
#Biru adalah data observaso
plt.scatter(x_test, y_test, color='blue')
#Merah adalah hasil prediksi dari machine learning
plt.plot(x_train,regressor.predict(x_train),color='red')
#judul dan label
plt.title('Salary vs Experience (Testing set)')
plt.xlabel('Years Experience')
plt.ylabel('Salary')
plt.show()
