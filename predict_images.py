import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd

# === Configuration ===
image_folder = "images"
model_path = "pool_classifier.pth"
output_csv = "predictions.csv"
class_names = ['no_pool', 'pool']
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# === Load Model ===
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Run Predictions ===
results = []

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
        predicted_class = class_names[predicted_idx]

    results.append({
        "filename": image_file,
        "prediction": predicted_class,
        "confidence": round(confidence, 4)
    })

# === Save to CSV ===
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved predictions to {output_csv}")
