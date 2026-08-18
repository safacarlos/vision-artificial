import os
import sys

# Ocultar logs de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow.lite as tflite

DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR_ACTUAL, "vww_96_grayscale_quantized.tflite")
LABELS_PATH = os.path.join(DIR_ACTUAL, "labels.txt")

if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] No se encuentra el archivo .tflite en: {MODEL_PATH}")
    sys.exit()

# Cargar modelo
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

try:
    clases = [line.strip() for line in open(LABELS_PATH, "r").readlines()]
except Exception:
    clases = ["0 papel y carton", "1 plastico", "2 metal"]

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("[ERROR] No se pudo acceder a la webcam.")
    sys.exit()

print("[ÉXITO] Modelo listo y cámara iniciada. Presioná 'Q' para salir.")

while True:
    exito, cuadro = camara.read()
    if not exito:
        break

    # 1. Escala de grises + Redimensionar a 96x96
    cuadro_gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
    imagen_96x96 = cv2.resize(cuadro_gris, (96, 96), interpolation=cv2.INTER_AREA)

    # 2. Convertir a float32 y normalizar entre 0.0 y 1.0
    matriz_imagen = imagen_96x96.astype(np.float32) / 255.0

    # 3. Formato final con dimensiones de batch y canal: (1, 96, 96, 1)
    matriz_imagen = np.expand_dims(matriz_imagen, axis=(0, -1))

    # 4. Inferencia
    interpreter.set_tensor(input_details[0]['index'], matriz_imagen)
    interpreter.invoke()
    prediccion = interpreter.get_tensor(output_details[0]['index'])[0]

    indice_maximo = np.argmax(prediccion)
    material_detectado = clases[indice_maximo]

    # Mostrar resultado
    cv2.putText(cuadro, f"IA: {material_detectado}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prueba IA - PC", cuadro)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()