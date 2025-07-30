import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as qq

# Inicializa o modelo YOLOv8 (pode ser yolov8n, yolov8s, etc.)
model = YOLO("yolov8n.pt")  # Baixa automaticamente se não existir

# Inicializa o tracker Deep SORT
tracker = DeepSort(max_age=30)

# Inicia captura de vídeo (0 para webcam ou caminho do vídeo)
cap = cv2.VideoCapture(0)

# Cores para identificação visual
np.random.seed(42)
colors = np.random.randint(0, 255, size=(100, 3), dtype="uint8")

print("[INFO] Iniciando detecção e rastreamento... Pressione Q para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv8 - detecção
    results = model.predict(frame, imgsz=640, conf=0.4)[0]

    detections = []
   # Filtrar apenas pessoas (classe 0)
    CLASSES_DESEJADAS = [0]  # só 'person'

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_id = int(class_id)

        if class_id in CLASSES_DESEJADAS:
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, class_id))


    # Deep SORT - rastreamento
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        class_id = track.get_det_class()
        x1, y1, x2, y2 = map(int, ltrb)

        color = [int(c) for c in colors[int(track_id) % len(colors)]]


        # Desenho da caixa e ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id} | Classe {class_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Exibe o frame
    cv2.imshow("YOLOv8 + Deep SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
