from ultralytics import YOLO
import cv2
from normalize import normalize_species

model = YOLO("runs/detect/train/weights/best.pt")

def predict_image(image_path):
    results = model(image_path)
    img = cv2.imread(image_path)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            raw_label = model.names[cls_id]
            species = normalize_species(raw_label)
            conf = box.conf[0].item()

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{species} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    cv2.imwrite("output.jpg", img)
    print("✅ Saved output.jpg")

if __name__ == "__main__":
    predict_image("test.jpg")
