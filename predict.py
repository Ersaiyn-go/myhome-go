from torchvision import models, transforms
from PIL import Image
import torch
import sys

# Загрузка классов
class_names = ['city', 'construction', 'mountain', 'park', 'water', 'yard']

# Трансформации
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Загрузка изображения
img_path = sys.argv[1]
image = Image.open(img_path).convert('RGB')
image = transform(image).unsqueeze(0)

# Загрузка модели
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Предсказание
with torch.no_grad():
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

print(f"Вид из окна: {class_names[predicted.item()]}")
