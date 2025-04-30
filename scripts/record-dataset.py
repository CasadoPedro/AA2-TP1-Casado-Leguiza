import cv2
import mediapipe as mp
import numpy as np
import os

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar imagen
image = cv2.imread("../hand.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gesto = "papel"
# Dataset filename
dataset_file = f"../dataset/{gesto}.npy"

# Inicializar la detección
with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            puntos = []
            for landmark in hand_landmarks.landmark:
                puntos.extend([landmark.x, landmark.y, landmark.z])  # Aplana (x, y, z)

            puntos = np.array(puntos)  # Ahora shape = (63,)

            # Si el dataset ya existe, lo cargamos y agregamos
            if os.path.exists(dataset_file):
                dataset = np.load(dataset_file)
                dataset = np.vstack([dataset, puntos])
            else:
                dataset = np.expand_dims(puntos, axis=0)  # Primera fila

            # Guardamos el dataset actualizado
            np.save(dataset_file, dataset)
            print(
                f"Guardada nueva fila en '{dataset_file}'. Dataset shape: {dataset.shape}"
            )

            # Dibujar la mano
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Mostrar imagen
cv2.imshow("Manos detectadas", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
