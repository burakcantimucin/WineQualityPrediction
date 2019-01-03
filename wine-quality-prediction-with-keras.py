# Veri önişleme ve analizi için kullanılacak kütüphane eklenir
import pandas as pd

# Kırmızı şarapların olduğu veri seti okunur
red_wine_dataset = pd.read_csv('../input/qualityofredwine/winequality-red.csv')

# Beyaz şarapların olduğu veri seti okunur
white_wine_dataset = pd.read_csv('../input/qualityofwhitewine/winequality-white.csv')

# Red şarapların olduğu veri setinin her bir sütununda toplam kaç tane boş kayıt olduğuna bakılır
red_wine_dataset.isnull().sum()

# Red şarapların olduğu veri setindeki tahmin edilecek değerin kaç farklı değer ve bu değerlerin toplam adetlerine bakılır
red_wine_dataset.quality.value_counts()

# Beyaz şarapların olduğu veri setinin her bir sütununda toplam kaç tane boş kayıt olduğuna bakılır
white_wine_dataset.isnull().sum()

# Beyaz şarapların olduğu veri setindeki tahmin edilecek değerin kaç farklı değer ve bu değerlerin toplam adetlerine bakılır
white_wine_dataset.quality.value_counts()

# Kırmızı ve beyaz şarapların olduğu veri setleri ayrı ayrı olarak parçalanır
# X değişkenlerinde veri setlerin öznitelikleri ve y değişkenlerinde de tahmin edilecek değer yer alır
red_wine_X = red_wine_dataset.iloc[:,:-1].values
red_wine_y = red_wine_dataset.iloc[:,-1].values

white_wine_X = white_wine_dataset.iloc[:,:-1].values
white_wine_y = white_wine_dataset.iloc[:,-1].values

# Kırmızı ve beyaz şarap veri setleri, eğitim ve test verileri olarak ayrılır
from sklearn.model_selection import train_test_split
red_wine_X_train, red_wine_X_test, red_wine_y_train, red_wine_y_test = train_test_split(red_wine_X, red_wine_y, test_size = 0.20, random_state = 4)
white_wine_X_train, white_wine_X_test, white_wine_y_train, white_wine_y_test = train_test_split(white_wine_X, white_wine_y, test_size = 0.20, random_state = 4)

# Tahminlerden sonra veri çatılarını birleştirmek için yedeklenir
cache_red_wine_X_test = red_wine_X_test
cache_red_wine_y_test = red_wine_y_test
cache_white_wine_X_test = white_wine_X_test
cache_white_wine_y_test = white_wine_y_test

# Öznitelikler arasındaki korelasyon sayıları düşük olduğundan veri setlerin öznitelikleri ölçeklendirilir
from sklearn.preprocessing import StandardScaler
red_wine_sc = StandardScaler()
red_wine_X_train = red_wine_sc.fit_transform(red_wine_X_train)
red_wine_X_test = red_wine_sc.transform(red_wine_X_test)
white_wine_sc = StandardScaler()
white_wine_X_train = white_wine_sc.fit_transform(white_wine_X_train)
white_wine_X_test = white_wine_sc.transform(white_wine_X_test)

# y değişkenlerindeki nümerik değerler kategorik hale dönüştürülür
red_wine_y_train = red_wine_y_train.astype('object')
red_wine_y_test = red_wine_y_test.astype('object')
white_wine_y_train = white_wine_y_train.astype('object')
white_wine_y_test = white_wine_y_test.astype('object')

# Kategorik veriler sayısallaştırılır
from sklearn.preprocessing import LabelEncoder
red_wine_labelencoder_y = LabelEncoder()
red_wine_y_train = red_wine_labelencoder_y.fit_transform(red_wine_y_train)
red_wine_y_test = red_wine_labelencoder_y.transform(red_wine_y_test)
white_wine_labelencoder_y = LabelEncoder()
white_wine_y_train = white_wine_labelencoder_y.fit_transform(white_wine_y_train)
white_wine_y_test = white_wine_labelencoder_y.transform(white_wine_y_test)

# Sayısal olan kategorik veriler ikili sisteme dönüştürülür
from keras.utils import np_utils
red_wine_y_train = np_utils.to_categorical(red_wine_y_train)
red_wine_y_test = np_utils.to_categorical(red_wine_y_test)
white_wine_y_train = np_utils.to_categorical(white_wine_y_train)
white_wine_y_test = np_utils.to_categorical(white_wine_y_test)

# Keras'ın model ve katman kütüphaneleri eklenir
from keras.models import Sequential
from keras.layers import Dense


# Kırmızı veri seti için Sequential nesnesinden değişken yaratılır ve değişkene 4 katman eklenir
red_wine_classifier = Sequential()
red_wine_classifier.add(Dense(units = 44, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
red_wine_classifier.add(Dense(units = 44, kernel_initializer = 'uniform', activation = 'relu'))
red_wine_classifier.add(Dense(units = 44, kernel_initializer = 'uniform', activation = 'relu'))
red_wine_classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))

# Model, ilgili parametreler ile derlenir
red_wine_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Bu örnek model eğitim ve test verileriyle eğitilir
red_wine_history = red_wine_classifier.fit(red_wine_X_train, red_wine_y_train, epochs = 200, batch_size = 128, verbose = 2, validation_data = (red_wine_X_test, red_wine_y_test))

# Beyaz veri seti için Sequential nesnesinden değişken yaratılır ve değişkene 4 katman eklenir
white_wine_classifier = Sequential()
white_wine_classifier.add(Dense(units = 44, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
white_wine_classifier.add(Dense(units = 44, kernel_initializer = 'uniform', activation = 'relu'))
white_wine_classifier.add(Dense(units = 44, kernel_initializer = 'uniform', activation = 'relu'))
white_wine_classifier.add(Dense(units = 7, kernel_initializer = 'uniform', activation = 'softmax'))

# Model, ilgili parametreler ile derlenir
white_wine_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Bu örnek model eğitim ve test verileriyle eğitilir
white_wine_history = white_wine_classifier.fit(white_wine_X_train, white_wine_y_train, epochs = 200, batch_size = 128, verbose = 2, validation_data = (white_wine_X_test, white_wine_y_test))

# Veri görselleştirmede kullanılan kütüphane eklenir
from matplotlib import pyplot as plt

# Kırmızı ve beyaz şarap için oluşturulan modellerin eğitim boyunca başarı oranını grafiksel olarak gösterir
plt.figure(figsize=(15,3))
plt.subplot(1, 2, 1)

plt.plot(red_wine_history.history['acc'])
plt.plot(red_wine_history.history['val_acc'])
plt.title('Red Wine Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')

plt.figure(figsize=(15,3))
plt.subplot(1, 2, 1)

plt.plot(white_wine_history.history['acc'])
plt.plot(white_wine_history.history['val_acc'])
plt.title('White Wine Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.show()

# Kırmızı şarap için eğitilen model tahminleme işlemine tabi tutulur
red_wine_y_pred = red_wine_classifier.predict(red_wine_X_test)

# Beyaz şarap için eğitilen model tahminleme işlemine tabi tutulur
white_wine_y_pred = white_wine_classifier.predict(white_wine_X_test)

# Kırmızı şarap tahminleri aşağıdaki döngüde 6'lık sisteme getirilir
red_wine_max_i = red_wine_y_pred.max(axis=1)
for i in range(len(red_wine_y_pred)):
    for j in range(6):
        if red_wine_y_pred[i,j] == red_wine_max_i[i]:
           red_wine_y_pred[i,j] = 1
        else:
           red_wine_y_pred[i,j] = 0
           
# Beyaz şarap tahminleri aşağıdaki döngüde 7'lik sisteme getirilir
white_wine_max_i = white_wine_y_pred.max(axis=1)
for i in range(len(white_wine_y_pred)):
    for j in range(7):
        if white_wine_y_pred[i,j] == white_wine_max_i[i]:
           white_wine_y_pred[i,j] = 1
        else:
           white_wine_y_pred[i,j] = 0
           
# Kırmızı şarap tahminlerinin toplam doğruluk oranı hesaplanır
red_wine_crt_values = (red_wine_y_pred == red_wine_y_test).sum()
red_wine_wrong_values = (red_wine_y_pred != red_wine_y_test).sum()
red_wine_total = red_wine_crt_values+red_wine_wrong_values
red_wine_result = red_wine_crt_values/red_wine_total
print(red_wine_result)

# Beyaz şarap tahminlerinin toplam doğruluk oranı hesaplanır
white_wine_crt_values = (white_wine_y_pred == white_wine_y_test).sum()
white_wine_wrong_values = (white_wine_y_pred != white_wine_y_test).sum()
white_wine_total = white_wine_crt_values+white_wine_wrong_values
white_wine_result = white_wine_crt_values/white_wine_total
print(white_wine_result)

# Karmaşıklık matrislerini bulmak adına kullanılan Python kütüphanesi eklenir
from sklearn.metrics import confusion_matrix
# Matematiksel işlemlerde kullanılan Python kütüphanesi eklenir
import numpy as np

# Kırmızı şarapların tahmin doğruluğunu görmek adına karmaşıklık matrisi oluşturulur
red_wine_y_test = [np.where(r==1)[0][0] for r in red_wine_y_test]
red_wine_y_pred = [np.where(r==1)[0][0] for r in red_wine_y_pred]
red_wine_cm = confusion_matrix(red_wine_y_test,red_wine_y_pred)
print(red_wine_cm)

# Beyaz şarapların tahmin doğruluğunu görmek adına karmaşıklık matrisi oluşturulur
white_wine_y_test = [np.where(r==1)[0][0] for r in white_wine_y_test]
white_wine_y_pred = [np.where(r==1)[0][0] for r in white_wine_y_pred]
white_wine_cm = confusion_matrix(white_wine_y_test, white_wine_y_pred)
print(white_wine_cm)


# Kırmızı ve beyaz şarap tahminleri eski tablolarla birleştirilir
red_wine_X_test = pd.DataFrame(cache_red_wine_X_test)
red_wine_y_test = pd.DataFrame(red_wine_y_test)
red_wine_y_pred = pd.DataFrame(red_wine_y_pred)

red_wine_X_test = pd.concat([red_wine_X_test, red_wine_y_test], axis=1)
red_wine_X_test = pd.concat([red_wine_X_test, red_wine_y_pred], axis=1)
red_wine_X_test.columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","real quality","predicted quality"]

white_wine_X_test = pd.DataFrame(cache_white_wine_X_test)
white_wine_y_test = pd.DataFrame(white_wine_y_test)
white_wine_y_pred = pd.DataFrame(white_wine_y_pred)
white_wine_X_test = pd.concat([white_wine_X_test, white_wine_y_test], axis=1)
white_wine_X_test = pd.concat([white_wine_X_test, white_wine_y_pred], axis=1)
white_wine_X_test.columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","real quality","predicted quality"]

red_wine_X_test.to_csv('prediction-of-red-wine-quality.csv')
white_wine_X_test.to_csv('prediction-of-white-wine-quality.csv')
