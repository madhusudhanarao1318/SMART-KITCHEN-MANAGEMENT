
# inference_real_time.py
import time
import argparse
import cv2
import torch
from utils import Inventory
from recipe_recommender import recommend_recipes

CLASS_NAMES = ['apple','banana','bottle','egg']

def load_model(weights_path=None, device=None):
    if weights_path:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    if device:
        model.to(device)
    model.conf = 0.4
    model.iou = 0.45
    return model

def run_video(source=0, weights=None):
    model = load_model(weights)
    inv = Inventory(deplete_seconds=60*20)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source {source}")
    last_recipe_print = 0
    recipe_interval = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            cls = int(cls)
            cls_name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
            detections.append((cls_name, float(conf), (x1,y1,x2,y2)))
            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        inv.update_from_detections(detections)
        inv.decay()

        summary = inv.get_inventory_summary()
        y = 30
        for k,v in summary.items():
            cv2.putText(frame, f"{k}: {v}", (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            y += 25

        cv2.imshow("Smart Kitchen - Detection", frame)

        now = time.time()
        if now - last_recipe_print > recipe_interval:
            recipes = recommend_recipes(summary)
            if recipes:
                print("Recipe suggestions based on current inventory:")
                for r in recipes:
                    print(" -", r)
            last_recipe_print = now

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0, help="Video source")
    parser.add_argument("--weights", default=None, help="Path to weights .pt")
    args = parser.parse_args()
    run_video(args.source, args.weights)
