
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet18(pretrained=True)
model.eval()

model.fc = torch.nn.Linear(model.fc.in_features, 2)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def cnn_predict(image_np):
    image = Image.fromarray(image_np)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return "real" if predicted.item() == 0 else "suspicious"