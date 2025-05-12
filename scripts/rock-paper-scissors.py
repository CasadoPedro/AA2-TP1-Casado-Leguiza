from tensorflow.keras.models import load_model
import cv2
import mediapipe as mp
import numpy as np

# Cargamos el modelo
model = load_model("../models/gesture_classifier.h5")

# Diccionario para mapear las clases a los gestos
clases = {0: "piedra", 1: "papel", 2: "tijera"}
# Capturamos una imagen
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
imagen_capturada = None
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar frame.")
        break

    # Mostrar la imagen en la ventana
    cv2.imshow("Camara", frame)

    # Si se presiona 'c', capturar la imagen
    if cv2.waitKey(1) & 0xFF == ord("c"):
        imagen_capturada = frame.copy()  # .copy() para guardar el frame actual
        print("📸 Foto capturada.")
        break
# Cerrar la captura de video
cap.release()
# Cerrar todas las ventanas
cv2.destroyAllWindows()
# Convertir la imagen capturada a RGB
imagen_rgb = cv2.cvtColor(imagen_capturada, cv2.COLOR_BGR2RGB)
# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
# Procesar la imagen con MediaPipe Hands
results = hands.process(imagen_rgb)
# Verificar si se detectó una mano
if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        puntos = []
        for landmark in hand_landmarks.landmark:
            puntos.extend([landmark.x, landmark.y])  # Aplana (x, y)

        puntos = np.array(puntos)  # Ahora shape = (42,) para la red neuronal

        # Predecir la clase de la imagen capturada
        prediccion = model.predict(np.expand_dims(puntos, axis=0))
        clase_predicha = np.argmax(prediccion, axis=1)[0]
        print(f"Clase predicha: {clases[clase_predicha]}")
else:
    print("No se detectó ninguna mano.")
