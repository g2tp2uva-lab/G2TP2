from ultralytics import YOLO
from roboflow import Roboflow
import os

# 1. Descargar el dataset desde Roboflow, en caso de no tenerlo en local 
#rf = Roboflow(api_key="bqp*******rny")        #indicar la API KEY de Roboflow
#project = rf.workspace("cocheconectado").project("data******0")        #indica el directorio del workspace
#version = project.version(7)
#dataset = version.download("yolov11")

# 2. Cargar tu modelo con los pesos .pt
model = YOLO('weights.pt')
path_yaml = 'Data*****0-6/data.yaml'        #indica aqui el path hasta el fichero data.yalm 

# 3. Ejecutar la validación en el set de TEST
# El archivo data.yaml descargado de Roboflow ya tiene las rutas configuradas
results = model.val(
    data=path_yaml,
    split='test',
    imgsz=640,  # Asegúrate de usar el mismo tamaño que en el entrenamiento
    save=True,  # Guarda los resultados visuales
    conf=0.25,  # Umbral de confianza (opcional)
    iou=0.6  # Umbral de Intersection over Union (opcional)
)

print("Validación completada.")
print(f"Los resultados, incluyendo la matriz de confusión, están en: {results.save_dir}")
