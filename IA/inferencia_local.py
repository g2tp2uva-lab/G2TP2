from pathlib import Path

import cv2
import json
from inference_sdk import InferenceHTTPClient
import supervision as sv


ROOT = Path(__file__).resolve().parent
IMAGE_FILE = ROOT / "train" / "fotos27abr" / "2026-04-27-101709.jpg"
ANNOTATED_DIR = ROOT / "anotated"
ROBOFLOW_API_KEY = "bqpX4QPrirFPhKE6irny"
MODEL_ID = "datasetp_v0/6"
API_URL = "https://serverless.roboflow.com"


def print_detected_classes(results: dict) -> None:
    predictions = results.get("predictions", [])
    if not predictions:
        print("No se detecto ninguna clase.")
        return

    print("Clases detectadas:")
    for prediction in predictions:
        detected_class = prediction.get("class", "desconocida")
        confidence = prediction.get("confidence")
        if confidence is None:
            print(f"- {detected_class}")
        else:
            print(f"- {detected_class} ({confidence:.3f})")


def filtrar_deteccion_mas_cercana(detecciones, umbral_confianza=0.80):
    # Filtrar por confianza mínima
    filtrados = [d for d in detecciones if d['confidence'] >= umbral_confianza]

    if not filtrados:
        return None

    # Encontrar la más "cercana" (la que tenga mayor área: width * height)
    # Usamos max() con una función lambda que calcula el área
    mas_cercana = max(filtrados, key=lambda d: d['width'] * d['height'])

    return mas_cercana


def main() -> None:
    image = cv2.imread(str(IMAGE_FILE))
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {IMAGE_FILE}")

    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=ROBOFLOW_API_KEY,
    )

    results = client.infer(str(IMAGE_FILE), model_id=MODEL_ID)
    print(results)
    print_detected_classes(results)
    detections = sv.Detections.from_inference(results)
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    ANNOTATED_DIR.mkdir(exist_ok=True)
    output_file = ANNOTATED_DIR / IMAGE_FILE.name
    cv2.imwrite(str(output_file), annotated_image)
    print(f"Imagen anotada guardada en: {output_file}")



    #Obtener la mas cercana
    predicciones = results.get('predictions', [])
    cercana = filtrar_deteccion_mas_cercana(predicciones)
    if cercana:
        print(f"Detección más cercana: {cercana['class']}")
        print(f"Confianza: {cercana['confidence']:.2f}")
        print(f"Dimensiones: {cercana['width']}x{cercana['height']}")
    else:
        print("No se encontraron detecciones con confianza suficiente.")

    sv.plot_image(annotated_image)


if __name__ == "__main__":
    main()
