from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the custom ResNet model (same as in your training script)
class CustomResNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=5):
        super(CustomResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# Load the trained model
model = CustomResNet(input_channels=3, num_classes=5)
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((180, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model(tensor)
    _, predicted = torch.max(outputs, 1)
    return predicted.item()

def get_rating(prediction):
    ratings = {
        1: "Poor",
        2: "Fair",
        3: "Good",
        4: "Very Good",
        5: "Excellent"
    }
    return ratings.get(prediction, "Unknown")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('file')
        predictions = []
        for file in files:
            if file:
                img_bytes = file.read()
                prediction = get_prediction(img_bytes)
                rating = get_rating(prediction + 1)  # Adjusting for 1-based index
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                with open(file_path, 'wb') as f:
                    f.write(img_bytes)
                predictions.append({'filename': file.filename, 'rating': rating, 'image_url': file.filename})
        return render_template('index.html', predictions=predictions)
    return render_template('index.html', predictions=None)

@app.route('/api/upload', methods=['POST'])
def api_upload():
    files = request.files.getlist('file')
    predictions = []
    for file in files:
        if file:
            img_bytes = file.read()
            prediction = get_prediction(img_bytes)
            rating = get_rating(prediction + 1)  # Adjusting for 1-based index
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            with open(file_path, 'wb') as f:
                f.write(img_bytes)
            predictions.append({'filename': file.filename, 'rating': rating, 'image_url': file.filename})
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)