# %%
import os
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_models


# %%
x = pd.read_csv("", index_col=0)
y = pd.read_csv("", index_col=0)

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# %%

pca = PCA(n_components=4)
x_pca = pca.fit_transform(x)


# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# %%
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.2, random_state=42)


# %%
svc = SVC(kernel='linear')

svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

svc_scores = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
]


rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

rfc_scores = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred)
]


# %%
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
x = np.arange(len(metrics))  
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - width/2, svc_scores, width, label="SVC", color="blue")
bars2 = ax.bar(x + width/2, rfc_scores, width, label="RFC", color="green")

# Etichette
ax.set_xlabel("Metriche")
ax.set_ylabel("Score")
ax.set_title("Comparison of SVC and Random Forest Performance")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

for bar in bars1 + bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f"{yval:.2f}", ha="center", va="bottom", fontsize=10)

plt.ylim(0, 1) 
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# %%

def f1_score(y_true, y_pred):
    precision = tf.keras.metrics.Precision()(y_true, y_pred)
    recall = tf.keras.metrics.Recall()(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# %%
model = Sequential([
    Dense(150, input_shape=(x_train.shape[1],), activation='relu'),
    Dropout(0.6),
    Dense(25, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])



optimizer = keras.optimizers.Adam(
        learning_rate=0.0008,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-10,
        amsgrad=False,
        weight_decay=0.01,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=None,
        name="adam",
    )

model.compile(loss='binary_crossentrpy', optimizer = optimizer, 
               metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), f1_score])



# %%
logs = model.fit(x_train, x_train, epochs=300, batch_size=32, validation_split = 0.2)
model.evaluate(x_test, y_test)


# %%

plt.plot(logs.history['accuracy'])
plt.plot(logs.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation F1 score
plt.plot(logs.history['f1_score'])
plt.plot(logs.history['val_f1_score'])
plt.title('Model F1 score')
plt.ylabel('F1 score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation recall
plt.plot(logs.history['recall'])
plt.plot(logs.history['val_recall'])
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation precision
plt.plot(logs.history['precision'])
plt.plot(logs.history['val_precision'])
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(logs.history['loss'])
plt.plot(logs.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()