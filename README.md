🍔 Food Classifier using GoogLeNet
This project is a binary image classifier that distinguishes between food and non-food images using a fine-tuned GoogLeNet (Inception v1) model. It includes:

Training and evaluation notebook (.ipynb)

Pre-trained model

Flask API for inference

Integrated dataset (Food-5K)

📁 Dataset
The dataset used is the Food-5K dataset, organized into:

mathematica
Copy
Edit
Food-5K/
├── training/
├── validation/
└── evaluation/
Each image is named using the format:

1_xxx.jpg — Food

0_xxx.jpg — Non-Food

🚀 Features
GoogLeNet fine-tuned on Food-5K for binary classification

Data augmentation and normalization

Visualizations for predictions and ROC curves

Flask REST API for inference on custom images

🛠️ Setup Instructions
1. Clone the repository
bash
Copy
Edit
git clone https://github.com/your-username/food-classifier.git
cd food-classifier
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
Key dependencies:

PyTorch

Torchvision

Flask

Matplotlib

scikit-learn

tqdm

Pillow

📚 Notebook Overview
Open the training notebook in Jupyter or Colab:

bash
Copy
Edit
jupyter notebook Food_Classifier_GoogLeNet.ipynb
Main components:

📦 FoodDataset: Custom torch.utils.data.Dataset for Food-5K

🧠 Model: Pretrained GoogLeNet with modified output layer

🔁 Data Augmentation: Resize, Random Flip, Color Jitter, etc.

📈 Training Loop: Accuracy, loss tracking, and model saving

📊 Evaluation: Classification report, confusion matrix, ROC curve

📷 Inference: predict_single_image() for new image predictions

✅ Model Performance
Dataset	Accuracy
Train	~99%
Validation	~95%
Test	~94%

Also included:

ROC Curve with AUC

Confusion matrix

Precision/Recall/F1-score metrics

🖥️ Flask API
The server.py file runs a lightweight API to serve the model:

🔧 Start the server
bash
Copy
Edit
python server.py
🖼️ Send a POST request
bash
Copy
Edit
curl -X POST http://127.0.0.1:5000/predict \
     -F "image=@/path/to/your/image.jpg"
📤 Response
json
Copy
Edit
{
  "class": "Food",
  "confidence": 97.53
}
🔍 Project Structure
bash
Copy
Edit
.
├── Food-5K/                  # Dataset
├── Food_Classifier_GoogLeNet.ipynb  # Training and evaluation notebook
├── server.py                 # Flask API
├── googlenet_food_best.pth  # Saved PyTorch model
└── README.md
📌 Notes
The model uses only the final classification layer for fine-tuning (rest frozen).

Inference can run on CPU; training was done using GPU (if available).

The dataset is small, so the model generalizes well with augmentations.

📬 Contact
Author: Tayyab Anees
Email: tayyabanees123321@gmail.com
