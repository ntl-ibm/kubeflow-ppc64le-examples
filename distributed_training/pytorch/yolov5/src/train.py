from ultralytics import YOLO
import yaml

cfg = yaml.safe_load("./data.yaml")

# Load a model
model = YOLO(cfg.get("model", "yolov8n.pt"))

# Train the model
results = model.train(data="./data.yaml", cfg="./train.cfg")

print(type(results))
print(str(results))
