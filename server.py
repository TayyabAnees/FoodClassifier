from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import io

app = Flask(__name__)

# Load pre-trained model
MODEL_PATH = "googlenet_food_best.pth"


def load_model():
    """Load trained GoogLeNet model"""
    model = models.googlenet(pretrained=False, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Modify last layer for 2 classes
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()  # Set model to evaluation mode
    return model


model = load_model()

# Define preprocessing for the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match GoogLeNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/predict', methods=['POST'])
def predict():
    """API route to predict food/non-food from an uploaded image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")  # Read and convert image

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    # Map predictions to class names
    class_names = ["Non-Food", "Food"]
    predicted_label = class_names[predicted_class.item()]
    confidence_score = confidence.item() * 100  # Convert to percentage

    # Return JSON response
    return jsonify({
        "prediction": predicted_label,
        "confidence": f"{confidence_score:.2f}%"
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
