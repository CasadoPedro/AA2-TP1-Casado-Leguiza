# Clasificador de Gestos: Piedra, Papel o Tijera✋✊✌️

## 📌 Descripción

Este proyecto implementa un sistema de clasificación de gestos para el juego "Piedra, Papel o Tijeras", utilizando visión por computadora y aprendizaje automático. La detección de la mano se realiza mediante **MediaPipe**, y la clasificación se realiza con una **red neuronal densa (MLP)** entrenada sobre coordenadas de puntos clave (landmarks) de la mano.

El proyecto está dividido en tres partes funcionales:

1. Grabación de dataset (`record-dataset.py`)
2. Entrenamiento del modelo (`train-gesture-classifier.py`)
3. Clasificación en tiempo real (`rock-paper-scissors.py`)


## 🎯 Objetivos

- Detectar gestos de mano usando **MediaPipe Hand Landmarker**
- Capturar y almacenar un dataset etiquetado de gestos
- Entrenar un clasificador utilizando una red neuronal
- Realizar inferencia en tiempo real con la cámara web


## 🧠 Tecnologías Utilizadas

- Python 3.10+
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- OpenCV
- TensorFlow / Keras
- NumPy
- Scikit-learn


## 📁 Estructura de Archivos
```
├── /scripts/  
    └──  record-dataset.py    
    └──  train-gesture-classifier.py
    └──  rock-paper-scissors.py  
├── /data/  
       └── dataset.npy  
├── /models/  
       └── gesture_classifier.h5  
└── /img-ejecucion/  
       └── capturas_del_funcionamiento.png
├── requirements.txt
└── README.md 
```

## 🧪 Scripts

### 1. `record-dataset.py`

Captura imágenes desde la cámara web, detecta landmarks con MediaPipe y guarda coordenadas `x,y` de 21 puntos (42 valores) más su etiqueta (`0`: piedra, `1`: papel, `2`: tijera) en un archivo `.npy`.

**Uso:**

```bash
python record-dataset.py piedra
```

### 🧪 2. `train-gesture-classifier.py` — Entrenamiento del Modelo

Este script entrena una red neuronal sobre los datos de landmarks capturados con MediaPipe. Separa el dataset en características (`X`) y etiquetas (`y`), realiza codificación one-hot de las clases y entrena usando validación con `EarlyStopping`.

#### 🧬 Arquitectura del modelo:

- Entrada: `42` neuronas (21 puntos x 2 coordenadas)
- Capas ocultas:
  - `Dense(128, relu)`
  - `Dense(64, relu)`
  - `Dense(32, relu)`
- Salida: `3` neuronas (`softmax`) para clasificar entre piedra, papel o tijera.

#### 📂 Dataset de entrada:

Debe estar guardado en `../data/dataset.npy` y debe haber sido generado previamente con `record-dataset.py`.

#### 💾 Modelo entrenado:

Se guarda como: `../models/gesture_classifier.h5`

#### ▶️ Cómo usar:

```bash
python train-gesture-classifier.py
```

### 🎮 3. `rock-paper-scissors.py` — Clasificador en Tiempo Real

Este script permite reconocer en tiempo real los gestos de "piedra", "papel" o "tijera" utilizando una imagen capturada desde la cámara web y un modelo previamente entrenado.

#### ⚙️ Funcionamiento

1. Se abre la cámara web y se muestra la imagen en pantalla.
2. Al presionar la tecla `c`, se captura una imagen.
3. Se procesan los landmarks de la mano utilizando **MediaPipe Hands**.
4. Se extraen los 21 puntos `(x, y)` y se convierten en un vector de 42 valores.
5. El modelo entrenado (`gesture_classifier.h5`) predice el gesto.
6. El resultado se muestra por consola como `"piedra"`, `"papel"` o `"tijera"`.

#### ▶️ Cómo usar

```bash
python rock-paper-scissors.py
```

## ⚠️ Advertencias

- Este proyecto fue desarrollado y probado con **Python 3.11**.  
  El comportamiento podría variar si se utiliza una versión diferente de Python, especialmente en lo que respecta a dependencias como `MediaPipe`, `TensorFlow` y `OpenCV`.

- La configuración de la cámara está seteada por defecto a **640x480** píxeles:
  ```python
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
  ```
  Esto fue elegido porque era la resolución disponible en el entorno de desarrollo, pero puede ser ajustado según las capacidades de la cámara del usuario.
