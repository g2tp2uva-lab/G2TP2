"""
Coche autónomo con seguimiento de línea + reacción a señales
=============================================================
Incluye 3 opciones de compensación de viraje por rueda defectuosa:
  - Opción A: Offset fijo
  - Opción B: Factor proporcional
  - Opción C: Autocompensación adaptativa
"""

import socket
import struct
import pickle
import cv2
import time
import threading
from collections import deque
import numpy as np
import cmath
import math

import artemis_autonomous_car

# ─── CONFIGURACIÓN DE RED ─────────────────────────────────────
SERVER_ADDRESS = ('172.16.0.1', 20003)
JETSON_IP      = '192.168.100.102'
JETSON_PORT    = 20004
RESULTS_PORT   = 20005

# ─── CONFIGURACIÓN DEL CONTROL AUTÓNOMO ──────────────────────
PATH                = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
CONFIDENCE_MIN      = 0.70
SIGNAL_TIMEOUT      = 2.0
THROTTLE_DEFAULT    = 0.65
THROTTLE_LIMITE_30  = 0.60
THROTTLE_LIMITE_90  = 0.75
THROTTLE_PELIGRO    = 0.45
STOP_DURATION       = 2.5
GIRO_FUERZA         = 0.8
GIRO_DURATION       = 1.0

# ─── COMPENSACIÓN DE VIRAJE (RUEDA DEFECTUOSA) ───────────────
# Activa SOLO UNA de las tres opciones poniéndola a True
USAR_OPCION_A = True    # Offset fijo (más simple)
USAR_OPCION_B = False   # Factor proporcional
USAR_OPCION_C = False   # Autocompensación adaptativa

# Opción A: Offset constante. Positivo = corrige hacia la izquierda
COMPENSACION_OFFSET = 0.23

# Opción B: Factor proporcional + offset
COMPENSACION_FACTOR = 1.15      # Multiplica el giro
COMPENSACION_OFFSET_B = 0.10    # Offset adicional

# Opción C: Autocompensación. Mantiene un histórico y se ajusta solo
HISTORICO_TAMANO = 30           # Frames a recordar para promediar
COMPENSACION_C_INICIAL = 0.10   # Offset inicial mientras se calibra
COMPENSACION_C_MAX = 0.40       # Límite del offset autocalculado
LEARNING_RATE = 0.05            # Velocidad de adaptación

# ─── CONFIGURACIÓN FRENADO PROGRESIVO STOP/PROHIBIDO-PASO ────
IMG_WIDTH  = 640
IMG_HEIGHT = 480
IMG_AREA   = IMG_WIDTH * IMG_HEIGHT
STOP_AREA_DETENIDO  = 1200.0
STOP_AREA_INICIO    = 500.0
THROTTLE_MIN_FRENO  = 0.5


# ─── ESTADO GLOBAL ────────────────────────────────────────────
ultimo_resultado_jetson = None
ultimo_timestamp_jetson = 0
ultima_senal_procesada  = None

estado_stop_hasta   = 0
estado_giro_hasta   = 0
direccion_giro      = 0
velocidad_actual    = THROTTLE_DEFAULT
frenando_por_senal  = False

# Estado para opción C (autocompensación adaptativa)
historico_giros = deque(maxlen=HISTORICO_TAMANO)
compensacion_aprendida = COMPENSACION_C_INICIAL


# ─── SOCKETS UDP ──────────────────────────────────────────────
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(SERVER_ADDRESS)
print(f"[CORE] Escuchando en {SERVER_ADDRESS}")

jetson_send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

results_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
results_sock.bind(('0.0.0.0', RESULTS_PORT))
results_sock.settimeout(0.01)


# ─── HILO QUE RECIBE RESULTADOS DE LA JETSON ─────────────────
def recibir_resultados_jetson():
    global ultimo_resultado_jetson, ultimo_timestamp_jetson
    while True:
        try:
            data, _ = results_sock.recvfrom(65535)
            if data[0:1] == b'Y':
                resultado = pickle.loads(data[1:])
                ultimo_resultado_jetson = resultado
                ultimo_timestamp_jetson = time.time()
        except socket.timeout:
            pass
        except Exception as e:
            print(f"[JETSON] Error: {e}")

threading.Thread(target=recibir_resultados_jetson, daemon=True).start()


# ─── ENVÍO DE CONTROL AL VEHÍCULO ────────────────────────────
def send_control(control_giro, control_acelerador, address):
    sock.sendto(
        struct.pack('c', bytes('C', 'ascii')) +
        struct.pack('d', round(control_giro, 3)) +
        struct.pack('d', round(control_acelerador, 3)),
        address
    )


# ─── COMPENSACIÓN DE VIRAJE ──────────────────────────────────
def compensar_viraje(giro_original):
    """
    Aplica compensación al giro según la opción activa.
    El coche vira a la derecha, así que sumamos un valor positivo
    para corregir hacia la izquierda.
    """
    global compensacion_aprendida, historico_giros

    if USAR_OPCION_A:
        # Offset fijo y constante
        return giro_original + COMPENSACION_OFFSET

    elif USAR_OPCION_B:
        # Factor proporcional + offset
        # Si giro original es 0, suma el offset
        # Si giro original es negativo (izquierda), lo amplifica
        # Si giro original es positivo (derecha), lo reduce
        if giro_original >= 0:
            return (giro_original / COMPENSACION_FACTOR) + COMPENSACION_OFFSET_B
        else:
            return (giro_original * COMPENSACION_FACTOR) + COMPENSACION_OFFSET_B

    elif USAR_OPCION_C:
        # Autocompensación adaptativa
        # Si el coche está intentando ir recto (giro original cerca de 0)
        # y el histórico muestra una tendencia, ajustamos
        historico_giros.append(giro_original)

        if len(historico_giros) >= HISTORICO_TAMANO:
            promedio = sum(historico_giros) / len(historico_giros)
            # Si el promedio es negativo, el coche está corrigiendo a la izquierda
            # constantemente, lo que significa que necesita más compensación
            if promedio < -0.05:
                compensacion_aprendida += LEARNING_RATE * 0.1
            elif promedio > 0.05:
                compensacion_aprendida -= LEARNING_RATE * 0.1

            # Limitar la compensación a un máximo
            compensacion_aprendida = max(0, min(COMPENSACION_C_MAX, compensacion_aprendida))

        return giro_original + compensacion_aprendida

    return giro_original


# ─── FRENADO PROGRESIVO ───────────────────────────────────────
def calcular_velocidad_frenado(area_senal):
    if area_senal <= STOP_AREA_INICIO:
        return THROTTLE_DEFAULT
    if area_senal >= STOP_AREA_DETENIDO:
        #time.sleep(0.3)
        return 0.0
    ratio = (area_senal - STOP_AREA_INICIO) / (STOP_AREA_DETENIDO - STOP_AREA_INICIO)
    velocidad = THROTTLE_DEFAULT * (1.0 - ratio)
    velocidad = max(velocidad, THROTTLE_MIN_FRENO) if ratio < 0.9 else max(velocidad, 0.0)
    return round(velocidad, 3)


# ─── LÓGICA DE REACCIÓN A SEÑALES ────────────────────────────
def aplicar_reaccion_senal(control_giro, control_acelerador):
    global ultima_senal_procesada
    global estado_stop_hasta, estado_giro_hasta, direccion_giro
    global velocidad_actual, frenando_por_senal

    ahora = time.time()

    if ahora < estado_stop_hasta:
        frenando_por_senal = False
        return (COMPENSACION_OFFSET, 0.0)

    # Justo al terminar el stop, restaurar velocidad normal
    if velocidad_actual == 0.0 and not frenando_por_senal:
        velocidad_actual = THROTTLE_DEFAULT
        ultima_senal_procesada = None
        print(f"[STOP] Reanudando marcha a velocidad {THROTTLE_DEFAULT}")

    if ahora < estado_giro_hasta:
        frenando_por_senal = False
        return (direccion_giro * GIRO_FUERZA, velocidad_actual)

    if ultimo_resultado_jetson is None:
        frenando_por_senal = False
        return (control_giro, velocidad_actual)

    if ahora - ultimo_timestamp_jetson > SIGNAL_TIMEOUT:
        frenando_por_senal = False
        return (control_giro, velocidad_actual)

    cercana = ultimo_resultado_jetson.get('mas_cercana')
    if cercana is None:
        frenando_por_senal = False
        return (control_giro, velocidad_actual)

    if cercana['confidence'] < CONFIDENCE_MIN:
        frenando_por_senal = False
        return (control_giro, velocidad_actual)

    clase = cercana['class']
    confianza = cercana['confidence']
    area_senal = cercana['width'] * cercana['height']

    if clase in ('STOP', 'PROHIBIDO-PASO'):
        velocidad_freno = calcular_velocidad_frenado(area_senal)
        velocidad_actual = velocidad_freno
        frenando_por_senal = True
        print(f"[FRENO] {clase} | área={area_senal:.0f}px² | velocidad={velocidad_freno:.2f}")
        if velocidad_freno == 0.0:
            print(f"[PARADO] Detenido ante {clase}")
            if clase == 'PROHIBIDO-PASO':
                estado_stop_hasta = ahora + 999999
                return (control_giro, 0.0)
            estado_stop_hasta = ahora + STOP_DURATION
            return (control_giro, 0.49)

    frenando_por_senal = False

    senal_id = (clase, round(cercana['x']), round(cercana['y']))
    if senal_id == ultima_senal_procesada:
        return (control_giro, velocidad_actual)

    ultima_senal_procesada = senal_id
    print(f"[ACCIÓN] Reaccionando a {clase} ({confianza:.2f})")

    if clase == 'LIMITE-30':
        velocidad_actual = THROTTLE_LIMITE_30
        return (control_giro, velocidad_actual)
    elif clase == 'LIMITE-90':
        velocidad_actual = THROTTLE_LIMITE_90
        return (control_giro, velocidad_actual)
    elif clase in ('PELIGRO-NINOS', 'PELIGRO-OPERARIO-OBRAS'):
        velocidad_actual = THROTTLE_PELIGRO
        return (control_giro, velocidad_actual)
    elif clase == 'OBLIGATORIO-GIRAR-IZQUIERDA':
        estado_giro_hasta = ahora + GIRO_DURATION
        direccion_giro = +1
        return (direccion_giro * GIRO_FUERZA, velocidad_actual)
    elif clase == 'OBLIGATORIO-GIRAR-DERECHA':
        estado_giro_hasta = ahora + GIRO_DURATION
        direccion_giro = -1
        return (direccion_giro * GIRO_FUERZA, velocidad_actual)
    elif clase == 'OBLIGATORIO-RECTO':
        return (control_giro, velocidad_actual)
    elif clase in ('CONO', 'OBSTACULO', 'VALLA'):
        velocidad_actual = THROTTLE_PELIGRO
        return (control_giro, velocidad_actual)

    return (control_giro, velocidad_actual)


# ─── INICIALIZAR EL CONTROL AUTÓNOMO ─────────────────────────
print("[CORE] Inicializando artemis_autonomous_car...")
aac = artemis_autonomous_car.artemis_autonomous_car(PATH)
aac.lidar_throttle_control = THROTTLE_DEFAULT

# Mostrar qué opción de compensación está activa
if USAR_OPCION_A:
    print(f"[COMPENSACIÓN] Opción A: Offset fijo = {COMPENSACION_OFFSET}")
elif USAR_OPCION_B:
    print(f"[COMPENSACIÓN] Opción B: Factor={COMPENSACION_FACTOR}, Offset={COMPENSACION_OFFSET_B}")
elif USAR_OPCION_C:
    print(f"[COMPENSACIÓN] Opción C: Autocompensación adaptativa")
else:
    print("[COMPENSACIÓN] Ninguna opción activa")


# ─── BUCLE PRINCIPAL ─────────────────────────────────────────
print("[CORE] Esperando frames del vehículo...")

while True:
    try:
        data, address = sock.recvfrom(99999)
    except KeyboardInterrupt:
        print("\n[CORE] Cerrando servidor.")
        break

    data_type = struct.unpack('c', bytes([data[0]]))[0]
    received_payload = bytes(data[1:])
    data_decoded = pickle.loads(received_payload, encoding='latin1')

    if data_type == b'I':
        img = cv2.imdecode(data_decoded, 1)
        if img is None:
            continue

        try:
            jetson_send_sock.sendto(data, (JETSON_IP, JETSON_PORT))
        except Exception as e:
            print(f"[JETSON] Error enviando frame: {e}")

        aac.lidar_throttle_control = velocidad_actual
        control_giro, control_acelerador, trayectory_not_found = \
            aac.proceso_fotograma(img, False, 0)

        if trayectory_not_found:
            control_giro = 0.0
            control_acelerador = velocidad_actual

        giro_final, acelerador_final = aplicar_reaccion_senal(
            control_giro, control_acelerador)

        # ── APLICAR COMPENSACIÓN DE VIRAJE ──────────────────
        giro_final = compensar_viraje(giro_final)
        # Limitar el giro al rango [-1, 1]
        giro_final = max(-1.0, min(1.0, giro_final))

        send_control(giro_final, acelerador_final, address)

        # ── MOSTRAR IMAGEN ──────────────────────────────────
        display_img = img.copy()

        if ultimo_resultado_jetson is not None and \
           time.time() - ultimo_timestamp_jetson < SIGNAL_TIMEOUT:
            cercana = ultimo_resultado_jetson.get('mas_cercana')
            if cercana and cercana['confidence'] >= CONFIDENCE_MIN:
                clase = cercana['class']
                confianza = cercana['confidence']
                area = cercana['width'] * cercana['height']
                texto = f"SENAL: {clase} ({confianza:.0%}) area={area:.0f}"
                overlay = display_img.copy()
                cv2.rectangle(overlay, (5, 5), (580, 50), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.5, display_img, 0.5, 0, display_img)
                color = (0, 0, 255) if clase in ('STOP', 'PROHIBIDO-PASO') else (0, 255, 0)
                cv2.putText(display_img, texto, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(display_img, "Sin senal detectada", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        else:
            cv2.putText(display_img, "Sin senal detectada", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        if ultimo_resultado_jetson is not None:
            todas = ultimo_resultado_jetson.get('detecciones', [])
            for i, det in enumerate(todas):
                texto_det = f"{det['class']} {det['confidence']:.0%}"
                cv2.putText(display_img, texto_det, (10, 70 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

        # Mostrar velocidad y compensación
        cv2.putText(display_img, f"Vel: {velocidad_actual:.2f} | Giro: {giro_final:.2f}",
                    (10, img.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if USAR_OPCION_C:
            cv2.putText(display_img, f"Comp.aprendida: {compensacion_aprendida:.3f}",
                        (10, img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Coche ARTEMIS", display_img)
        cv2.waitKey(1)

    elif data_type == b'L':
        pass
    elif data_type == b'D':
        pass


cv2.destroyAllWindows()
sock.close()
jetson_send_sock.close()
results_sock.close()
