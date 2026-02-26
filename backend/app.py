from fastapi import FastAPI, UploadFile
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import io
import csv

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchvision.models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load("model/waste_model.pth", map_location=device))
model.eval()

classes = ["hazardous", "non_hazardous", "organic", "recyclable"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile):
    image = Image.open(io.BytesIO(await file.read()))
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, 1).item()

    predicted_class = classes[pred]

    # Append to log file for Pathway
    with open("waste_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([predicted_class])

    return {"class": predicted_class}