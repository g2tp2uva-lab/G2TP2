import socket
import struct
import pickle
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ─── CONFIGURACIÓN ────────────────────────────────────────────
CORE_IP 	 = '192.168.100.101'   # IP del Core
RESULTS_PORT     = 20005            # Puerto donde el Core recibe resultados
LISTEN_PORT      = 20004            # Puerto donde la Jetson escucha frames
MODEL_PATH       = Path.home() / "weights.pt"
CONFIDENCE_THRESHOLD = 0.70

# ─── CARGAR MODELO ────────────────────────────────────────────
print("Cargando modelo YOLOv11...")
model = YOLO(str(MODEL_PATH))
print("Modelo listo. Escuchando frames del Core...")

# ─── SOCKETS ──────────────────────────────────────────────────
recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
recv_sock.bind(('0.0.0.0', LISTEN_PORT))

send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ─── BUCLE PRINCIPAL ─────────────────────────────────────────
while True:
    data, addr = recv_sock.recvfrom(99999)

    # Verificar que es un frame de imagen (tipo 'I')
    data_type = struct.unpack('c', bytes([data[0]]))[0]
    if data_type != b'I':
        continue

    # Decodificar imagen (mismo formato que el Core)
    payload = pickle.loads(bytes(data[1:]), encoding='latin1')
    buf = np.frombuffer(payload, dtype=np.uint8) if isinstance(payload, bytes) else payload
    frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    if frame is None:
        continue

    # ── INFERENCIA YOLOV11 EN GPU ────────────────────────────
    results = model(frame, device=0, conf=0.5, verbose=False)

    # ── EXTRAER DETECCIONES ──────────────────────────────────
    detecciones = []
    names = results[0].names
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detecciones.append({
            'class':      names[int(box.cls)],
            'confidence': float(box.conf),
            'x':          (x1 + x2) / 2,
            'y':          (y1 + y2) / 2,
            'width':      x2 - x1,
            'height':     y2 - y1,
        })

    # ── FILTRAR Y ENCONTRAR LA MÁS CERCANA ───────────────────
    filtradas = [d for d in detecciones if d['confidence'] >= CONFIDENCE_THRESHOLD]
    cercana = max(filtradas, key=lambda d: d['width'] * d['height']) if filtradas else None

    if cercana:
        print(f"Señal: {cercana['class']} ({cercana['confidence']:.2f})")
        
  
    # ── ENVIAR RESULTADO AL CORE ─────────────────────────────
    resultado = {
        'detecciones': detecciones,
        'mas_cercana': cercana    
    }
    payload_out = b'Y' + pickle.dumps(resultado)
    #print(f"[DEBUG] Tamaño paquete a enviar: {len(payload_out)} bytes")
    send_sock.sendto(payload_out, (CORE_IP, RESULTS_PORT))
    
    
    
    
    
