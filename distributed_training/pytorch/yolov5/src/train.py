from ultralytics import YOLO
import yaml

with open("./data.yaml") as f:
    cfg = yaml.safe_load(f)

# Load a model
model = YOLO(cfg.get("model", "yolov8n.pt"))

# Train the model
results = model.train(data="./data.yaml", cfg="./train.yaml")

print(type(results))
print(str(results))
