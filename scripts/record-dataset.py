import cv2
import mediapipe as mp
import numpy as np
import os
import time
import sys

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Verificar que el argumento del gesto haya sido pasado
if len(sys.argv) < 2:
    print("Por favor, pasa un argumento para el gesto ('papel', 'piedra', 'tijera').")
    sys.exit(1)

gesto = sys.argv[1]
dataset_file = "../data/dataset.npy"  # Guardamos los datos en un archivo .npy

# Codificación de gestos
gesto_to_label = {"piedra": 0, "papel": 1, "tijera": 2}

# Verificar que el gesto esté entre las opciones válidas
if gesto not in gesto_to_label:
    print("Gesto no válido. Elige entre 'piedra', 'papel', o 'tijera'.")
    sys.exit(1)

label = gesto_to_label[gesto]  # Asignamos la etiqueta correspondiente

# Inicializar la detección
with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    # Iniciar la captura de video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    captured_images = []
    interval = 3  # segundos entre captura de imágenes
    start_time = time.time()

    print("Presioná 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar frame.")
            break

        # Mostrar la imagen en la ventana
        cv2.imshow("Camara", frame)

        # Si pasaron 3 segundos desde la última captura
        if time.time() - start_time >= interval:
            captured_images.append(frame.copy())  # .copy() para guardar el frame actual
            print(f"📸 Foto capturada. Total guardadas: {len(captured_images)}")
            start_time = time.time()  # reiniciar el contador

        # Si se presiona 'q', salir del bucle
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Procesar todas las imágenes capturadas con MediaPipe
    for img in captured_images:
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                puntos = []
                for landmark in hand_landmarks.landmark:
                    puntos.extend([landmark.x, landmark.y])  # Aplana (x, y)

                puntos = np.array(puntos)  # Ahora shape = (42,)

                # Añadir la etiqueta (label) al final de los puntos
                puntos = np.append(puntos, label)

                # Si el dataset ya existe, lo cargamos y agregamos
                if os.path.exists(dataset_file):
                    dataset = np.load(dataset_file)
                    dataset = np.vstack([dataset, puntos])
                else:
                    dataset = np.expand_dims(puntos, axis=0)  # Si no existe lo creamos

                # Guardamos el dataset actualizado
                np.save(dataset_file, dataset)
                print(
                    f"Guardada nueva fila en '{dataset_file}'. Dataset shape: {dataset.shape}"
                )

                # Dibujar la mano sobre la imagen original
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Mostrar la imagen con los puntos de la mano
                cv2.imshow("Mano detectada", img)
                cv2.waitKey(0)  # Esperar una tecla para pasar a la siguiente imagen

    cap.release()
    cv2.destroyAllWindows()
