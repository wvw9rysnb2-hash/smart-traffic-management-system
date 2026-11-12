from ultralytics import YOLO

model = YOLO("best.pt")

print("Model class names:")
print(model.names)

print("\nClass names one by one:")
for class_id, class_name in model.names.items():
    print(f"ID {class_id}: {class_name}")