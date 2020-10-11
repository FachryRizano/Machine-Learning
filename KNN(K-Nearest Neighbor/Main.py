#Data training : sejumlah data mahasiswa yang lulus tepat
#waktu dan yang tidak, berdasarkan IPK S1 dan Usia
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets #memanggil library sklearn untuk import dataset
from sklearn.neighbors import KNeighborsClassifier #memanggil library sklearn untuk KNN Clasifier

#Memanggil IRIS dataset dari objek skrlearn
#memasukkannya ke dalam variable bunga
bunga = datasets.load_iris()
#menampilkan tipe obyek dari bung
print(type(bunga))
#menampilkan jumlah baris dan kolom dari dataset
print(bunga.data.shape)
#menampilkan target set dari data
print(bunga.target_names)

#memanggil training dataset
X = bunga.data
#memanggil target set
Y = bunga.target_names
#Melakukan konversi tipe datasets'ke dalam tipe dataframe
df = pd.DataFrame(X, columns=bunga.feature_names)
#mencetak 5 data pertama dari dataframe
print(df.head())

#Memanggil KNN classifier
knn = KNeighborsClassifier(n_neighbors=6,weights='uniform',algorithm='auto',metric='euclidean')

#Fitting model dengan training data and target
X_train = bunga['data']
Y_train = bunga['target']
knn.fit(X_train, Y_train)

#CONTOH Melakukan Prediksi/Klasifikasi
#Data yang akan diprediksi
Data = [[6.2,1.5,4.2,2.6]]

#melakukan prediksi berdasarkan
Y_pred = knn.predict(Data)

#Mencetak hasil prediksi
# Hasil 0 adalah setosa
# Hasil 1 adalah Versicolor
# Hasil 2 adalah Verginica
print("Hasil Prediksi: Jenis bunga", Y_pred)
print(type(bunga))