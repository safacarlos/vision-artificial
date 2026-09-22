import os
import sys
import time
from collections import Counter

# Ocultar advertencias de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
import tensorflow.lite as tflite

DIR_ACTUAL = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DIR_ACTUAL, "vww_96_grayscale_quantized.tflite")
LABELS_PATH = os.path.join(DIR_ACTUAL, "labels.txt")

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

# Variables de control
NUM_MUESTRAS = 10
TIEMPO_ENTRE_FOTOS = 0.3  # Pausa en segundos entre cada foto (ajustable)
resultado_final = "Esperando analisis..."

print("[ÉXITO] Presioná 'ESPACIO' para evaluar o 'Q' para salir.")

while True:
    exito, cuadro = camara.read()
    if not exito:
        break

    # Mostrar en pantalla
    cv2.putText(cuadro, f"Resultado: {resultado_final}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(cuadro, "Presiona ESPACIO para evaluar", (20, 450), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Prueba IA - Votacion", cuadro)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord(' '):
        predicciones_acumuladas = []
        print(f"\n[ANALIZANDO] Tomando {NUM_MUESTRAS} muestras pausadas...")

        for i in range(1, NUM_MUESTRAS + 1):
            exito_muestra, cuadro_muestra = camara.read()
            if not exito_muestra:
                continue

            # Mostrar mensaje de progreso en pantalla mientras saca fotos
            cuadro_progreso = cuadro_muestra.copy()
            cv2.putText(cuadro_progreso, f"Capturando foto {i}/{NUM_MUESTRAS}...", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Prueba IA - Votacion", cuadro_progreso)
            cv2.waitKey(1)

            # Preprocesamiento
            cuadro_gris = cv2.cvtColor(cuadro_muestra, cv2.COLOR_BGR2GRAY)
            imagen_96x96 = cv2.resize(cuadro_gris, (96, 96), interpolation=cv2.INTER_AREA)
            matriz_imagen = imagen_96x96.astype(np.float32) / 255.0
            matriz_imagen = np.expand_dims(matriz_imagen, axis=(0, -1))

            # Inferencia
            interpreter.set_tensor(input_details[0]['index'], matriz_imagen)
            interpreter.invoke()
            prediccion = interpreter.get_tensor(output_details[0]['index'])[0]

            indice = np.argmax(prediccion)
            predicciones_acumuladas.append(clases[indice])

            # Pausa para dar tiempo entre capturas
            time.sleep(TIEMPO_ENTRE_FOTOS)

        # Contar votos
        conteo = Counter(predicciones_acumuladas)
        resultado_final, votos = conteo.most_common(1)[0]

        print(f"[DESGLOSE DE VOTOS] {dict(conteo)}")
        print(f"[GANADOR CONFIRMADO] {resultado_final} ({votos}/{NUM_MUESTRAS} coincidencias)\n")

    elif tecla == ord('q'):
        break

camara.release()
cv2.destroyAllWindows()