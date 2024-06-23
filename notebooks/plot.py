import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
# from kerastuner.tuners import RandomSearch
from scipy.signal import savgol_filter
from mlxtend.evaluate import confusion_matrix
import random

data = pd.read_csv("/home/sonn/Son/Workspace/IR_PhanLoaiNuocCam/IR/data/processed/data_giong.csv")
data.head()

X = data.iloc[:,1:]
# X.head()

# tenmau = data["TenMau"]

# ma = []
# for mau in tenmau:
#     ma.append(mau.replace("/", "")[0:2])


# data["TenMau"] = ma

# data.head()

# data2 = data.drop(['Ma'], axis=1)

# data2.head()

y = data['TenMau']

label_counts = y.value_counts()

# Plotting
plt.bar(label_counts.index, label_counts.values)
plt.xlabel('Nhãn')
plt.ylabel('Số mẫu')
plt.show()

labels = np.unique(y)
le = LabelEncoder()
y = le.fit_transform(y)

X = savgol_filter(X, window_length=25, polyorder=3, deriv=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# from imblearn.over_sampling import SMOTE
# # Resampling the minority class. The strategy can be changed as required.
# sm = SMOTE()
# # Fit the model to generate the data.
# X_train, y_train = sm.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = TSNE(3)
X_train_pca = pca.fit_transform(X_train)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c = y_train)
ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], X_train_pca[:, 2], c=y_train)
plt.show()
