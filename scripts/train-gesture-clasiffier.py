import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


# Cargar nuestro dataset
data = np.load("../data/dataset.npy")
# Separar el dataset en características (X) y etiquetas (y)
y = data[:, -1]
X = data[:, :-1]
# Codificar las etiquetas para poder usarlas en la red neuronal
y = to_categorical(y, num_classes=len(np.unique(y)))
# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Definimos la arquitectura de la red neuronal
model = Sequential(
    [
        Input(shape=(42,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(3, activation="softmax"),  # 3 clases: piedra, papel, tijeras
    ]
)


# Compilamos el modelo
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Usamos early stopping para frenar el entrenamiento
early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode="min")

# Entrenar el modelo
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    callbacks=[early_stopping],
    batch_size=32,
)

# Guardar el modelo
model.save("../models/gesture_classifier.h5")
